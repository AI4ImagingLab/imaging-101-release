"""Standalone field/nuisance evaluator and independent NumPy physical forward check.

No solver, compiler, upstream package, submitted code, or cached prediction is
imported. The Green-boundary Fourier integral independently evaluates the same
piecewise-flat facet acquisition as the selected analytic reconstruction.
"""
from pathlib import Path
import argparse,json,math,sys
import numpy as np
from scipy.special import lpmv

HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,402,3),'spin':(1,3),'ao_offsets':(1,2,2),'ao_log_gains':(1,2),'occultation_offset':(1,1,2)}

def _rotation(spin,time):
    beta=(90-spin[0])*np.pi/180;longitude=spin[1]*np.pi/180;phase=48*np.pi/spin[2]*np.asarray(time)
    cb,sb=np.cos(beta),np.sin(beta);cl,sl=np.cos(longitude),np.sin(longitude)
    pole=np.array([[cb*cl,cb*sl,-sb],[-sl,cl,0.],[sb*cl,sb*sl,cb]])
    f=np.zeros(np.shape(phase)+(3,3));f[...,0,0]=np.cos(phase);f[...,0,1]=np.sin(phase);f[...,1,0]=-np.sin(phase);f[...,1,1]=np.cos(phase);f[...,2,2]=1
    return f@pole

def _frame(observer,up):
    z=observer/np.linalg.norm(observer);x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x);y/=np.linalg.norm(y)
    return np.stack([x,y,z])

def _mesh_geometry(mesh,faces):
    triangle=mesh[faces];cross=np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]);area=np.linalg.norm(cross,axis=1)/2
    if np.any(area<=0):raise ValueError('Degenerate submitted mesh facets')
    normal=cross/(2*area[:,None]);centroid=triangle.mean(axis=1)
    delta=triangle[None,:,:,:]-centroid[:,None,None,:]
    above=np.einsum('abvd,ad->abv',delta,normal)>0
    facing=np.einsum('abvd,bd->abv',delta,normal)<0
    shared=np.any(faces[None,:,None,:]==faces[:,None,:,None],axis=2)
    potential=np.any(above&facing&~shared,axis=2);np.fill_diagonal(potential,False)
    return triangle,normal,area,centroid,potential

def _visibility(geometry,observer,sun):
    triangle,normal,area,centroid,potential=geometry
    observer=np.atleast_2d(observer);sun=np.atleast_2d(sun)
    mu=observer@normal.T;mu0=sun@normal.T;visible=(mu>=0)&(mu0>=0)
    for face in range(len(triangle)):
        blocks=np.flatnonzero(potential[face])
        if not len(blocks):continue
        t=triangle[blocks];s1=t[:,0]-t[:,1];s2=t[:,0]-t[:,2];b=t[:,0]-centroid[face]
        determinant_vector=np.cross(s1,s2);gamma_vector=np.cross(s1,b);beta_vector=np.cross(b,s2)
        for direction,dots in [(observer,mu),(sun,mu0)]:
            determinant=direction@determinant_vector.T
            with np.errstate(divide='ignore',invalid='ignore'):
                gamma=(direction@gamma_vector.T)/determinant;beta=(direction@beta_vector.T)/determinant
            intersects=(abs(determinant)>=1e-6)&(gamma>=0)&(gamma<=1)&(beta>=0)&(beta<=1-gamma)&(dots[:,blocks]<0)
            visible[:,face]&=~intersects.any(axis=1)
    return visible,mu,mu0

def _image_fourier(image):
    image=np.maximum(image,0);z=np.fft.rfft2(image);y,x=np.indices(z.shape);z=z*((-1.)**(x+y))/z[0,0].real
    return z.ravel()[1:]

def _harmonics(vertex_angles):
    theta,phi=vertex_angles.T;basis=np.empty((402,100))
    for l in range(10):
        for m in range(-l,l+1):
            k=abs(m);value=np.sqrt((2*l+1)*math.factorial(l-k)/(4*np.pi*math.factorial(l+k)))*lpmv(k,l,np.cos(theta))
            if m:value*=np.sqrt(2)*(np.cos(k*phi) if m>0 else np.sin(k*phi))
            basis[:,l*(l+1)+m]=value
    return basis

def forward_objective(output,raw,return_details=False):
    """Full native weighted sum of squares, using an independent direct forward.

    Three harmonics blocks are inferred from the submitted ordered mesh by
    fixed representation equations; no hidden coefficient array is consulted.
    """
    data={key:value[0] for key,value in raw.items()};mesh=output['reconstruction'][0];spin=output['spin'][0];faces=data['facets'].astype(int)
    geometry=_mesh_geometry(mesh,faces);triangle,normal,area,centroid,potential=geometry
    time=data['lc_time'].ravel();rotation=_rotation(spin,time)
    observer=np.einsum('nij,nj->ni',rotation,data['lc_observer']);sun=np.einsum('nij,nj->ni',rotation,data['lc_sun'])
    visible,mu,mu0=_visibility(geometry,observer,sun)
    denominator=mu+mu0
    with np.errstate(divide='ignore',invalid='ignore'):brightness=np.where(visible,mu*mu0*(1/denominator+.1),0.)@area
    for curve in range(30):
        rows=data['lc_curve_index']==curve
        if data['lc_relative'][curve]:brightness[rows]/=brightness[rows].mean()
    lc_residual=8*(data['lc_brightness'].ravel()-brightness)
    ny,nx=150,150;u,v=np.meshgrid(np.arange(76)/(150*.009942),np.fft.fftfreq(150,d=.009942));frequency=np.stack([u.ravel()[1:],v.ravel()[1:]],axis=1)
    ao_models=[];ao_residual=[]
    for exposure in range(2):
        rotation=_rotation(spin,data['ao_time'][exposure,0]);observer=rotation@data['ao_observer'][exposure];sun=rotation@data['ao_sun'][exposure]
        visible,mu,mu0=_visibility(geometry,observer,sun);ids=np.flatnonzero(visible[0]);mu=mu[0,ids];mu0=mu0[0,ids]
        intensity=mu0*(1/(mu+mu0)+.1);scale=206264.80624709636/(149597871.*data['ao_distance'][exposure,0])
        transform=_frame(data['ao_observer'][exposure],data['ao_up'][exposure])@rotation.T
        projected=(triangle[ids]@transform.T)*scale
        total=(intensity*mu*area[ids]).sum()*scale**2;model=np.zeros(len(frequency),dtype=np.complex128)
        for start in range(0,len(frequency),256):
            f=frequency[start:start+256];value=np.zeros((len(f),len(ids)),dtype=np.complex128)
            for edge in range(3):
                a=projected[:,edge,:2];b=projected[:,(edge+1)%3,:2];delta=b-a
                q=f@delta.T;phase=f@((a+b)/2).T
                cross=f[:,0,None]*delta[None,:,1]-f[:,1,None]*delta[None,:,0]
                value+=1j/(2*np.pi)*cross/(f[:,0]**2+f[:,1]**2)[:,None]*np.exp(-2j*np.pi*phase)*np.sinc(q)
            model[start:start+len(f)]=(value@intensity)/total
        model*=np.exp(2j*np.pi*(frequency@output['ao_offsets'][0,exposure]))
        model*=np.exp(output['ao_log_gains'][0,exposure])*_image_fourier(data['ao_psf'][exposure]);ao_models.append(model)
        ao_residual.append(2*(_image_fourier(data['ao_images'][exposure])-model))
    # The author acquisition has only detected chords; retain every endpoint.
    if np.any(data['oc_chord_type']<0):raise ValueError('Unsupported changed occultation acquisition')
    transform=_frame(data['oc_observer'][0],data['oc_up'][0])@_rotation(spin,data['oc_time'].mean()).T
    xy=(mesh@transform.T)[:,:2]+output['occultation_offset'][0,0]
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    first=xy[edges[:,0]];last=xy[edges[:,1]];segment=last-first;oc_residual=[];endpoints=[];missed_chords=0
    for chord in data['oc_chords']:
        a,b=chord[:2],chord[2:];direction=b-a
        denominator=direction[0]*(first[:,1]-last[:,1])-direction[1]*(first[:,0]-last[:,0])
        with np.errstate(divide='ignore',invalid='ignore'):fraction=((a[0]-first[:,0])*direction[1]-(a[1]-first[:,1])*direction[0])/denominator
        allowed=(abs(denominator)>=1e-10)&(fraction>=0)&(fraction<=1)
        intersections=first[allowed]+fraction[allowed,None]*segment[allowed]
        if len(intersections)<2:
            fallback,_=_missing_chord_residual(xy,chord)
            oc_residual.extend(.1*fallback);endpoints.append(np.full(4,np.nan));missed_chords+=1
            continue
        along=intersections@direction;entry=intersections[np.argmin(along)];exit=intersections[np.argmax(along)]
        endpoints.append(np.r_[entry,exit]);oc_residual.extend(.1*np.r_[entry-a,exit-b])
    # Directed adjacent concavities and the selected undirected dihedral sum.
    adjacent=np.zeros_like(potential)
    for a in range(len(faces)):
        adjacent[a]=(faces[a,None,:,None]==faces[:,None,:]).any(axis=2).sum(axis=1)>=2
    convex=float(np.sum(np.where(potential&adjacent,area[None,:]*(1-normal@normal.T),0))/area.sum())
    edge_faces={}
    for face,vertices in enumerate(faces):
        for a,b in [(vertices[0],vertices[1]),(vertices[1],vertices[2]),(vertices[2],vertices[0])]:edge_faces.setdefault(tuple(sorted((int(a),int(b)))),[]).append(face)
    dihedral=sum((1-float(normal[pair[0]]@normal[pair[1]]))**2 for pair in edge_faces.values() if len(pair)==2)
    determinant=np.einsum('ij,ij->i',triangle[:,0],np.cross(triangle[:,1],triangle[:,2]));volume=determinant.sum()/6
    com=np.linalg.norm((determinant[:,None]*triangle.sum(axis=1)).sum(axis=0)/(24*volume))
    basis=_harmonics(data['vertex_angles']);theta,phi=data['vertex_angles'].T
    factors=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],axis=1);coefficients=[]
    for axis in range(3):
        rows=abs(factors[:,axis])>1e-12;ratio=mesh[rows,axis]/factors[rows,axis]
        if np.any(ratio<=0):raise ValueError('Mesh violates positive radial harmonic representation')
        coefficients.append(np.linalg.lstsq(basis[rows],np.log(ratio),rcond=None)[0])
    b=coefficients[1]-coefficients[0];c=coefficients[2]-coefficients[0];degree=np.concatenate([np.full(2*l+1,l/9) for l in range(10)])
    octantoid=float(np.sum(degree*(b*b+c*c)))
    penalties=np.array([30*convex,5*octantoid,.5*dihedral,com])
    components=dict(lightcurves=float(lc_residual@lc_residual),adaptive_optics=float(sum(np.vdot(r,r).real for r in ao_residual)),occultation=float(np.dot(oc_residual,oc_residual)),regularizers=float(penalties@penalties))
    objective=float(sum(components.values()))
    if not np.isfinite(objective):raise ValueError('Nonfinite physical objective')
    if return_details:return objective,dict(components=components,lightcurve_model=brightness,ao_models=np.array(ao_models),occultation_endpoints=np.array(endpoints),missed_chord_count=missed_chords,penalties=penalties)
    return objective


def _missing_chord_residual(xy,chord):
    """Native positive-chord optimization penalty, not physical endpoints.

    xy already includes the fitted sky-plane offset. The signed support-line
    distance follows the source extreme/nearest-side rule. Source k=4 enters
    exp(-2*k*distance), so the fixed sigmoid steepness is 8 inverse kilometres. No observed endpoint is discarded.
    """
    from scipy.special import expit
    a,b=chord[:2],chord[2:]
    direction=b-a;direction=direction/np.linalg.norm(direction)
    normal=np.array([-direction[1],direction[0]])
    distance=(a-xy)@normal
    negative=distance[distance<0];positive=distance[distance>0]
    if len(negative) and len(positive):
        low=float(negative.min());high=float(positive.max())
        signed=low if -low<=high else -high
    else:
        signed=min(float(-negative.max()) if len(negative) else 1e6,float(positive.min()) if len(positive) else 1e6)
    sigmoid=float(expit(8*signed))
    return np.array([a[0]*(1+sigmoid),a[1],b[0],b[1]]),sigmoid


def _read_npz(path):
    path=Path(path)
    if not path.is_file():raise ValueError(f'NPZ does not exist: {path}')
    with np.load(path,allow_pickle=False) as archive:
        result={}
        for key,shape in SHAPES.items():
            if key not in archive:raise ValueError(f'Missing required key: {key}')
            value=archive[key]
            if value.shape!=shape:raise ValueError(f'Invalid shape for {key}: {value.shape}')
            if value.dtype.kind!='f' or value.dtype.itemsize not in (4,8):raise ValueError(f'{key} must be real float32/float64')
            if not np.isfinite(value).all():raise ValueError(f'Nonfinite {key}')
            result[key]=value.astype(np.float64)
    return result

def field_metrics(prediction,reference):
    metrics={}
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        for axis,label in enumerate('xyz'):
            dynamic=np.ptp(reference['reconstruction'][...,axis])
            if dynamic<=0 or not np.isfinite(dynamic):raise ValueError('Invalid zero-range reference coordinate')
            metrics['mesh_'+label+'_range_nrmse']=float(np.sqrt(np.mean((prediction['reconstruction'][...,axis]-reference['reconstruction'][...,axis])**2))/dynamic)
        for i,name in enumerate(['latitude_deg','longitude_deg','period_hours']):
            difference=float(prediction['spin'][0,i]-reference['spin'][0,i])
            if i==1:difference=(difference+180)%360-180
            metrics[name+'_absolute_error']=abs(difference)
        for key,unit in [('ao_offsets','arcsec'),('ao_log_gains','dimensionless'),('occultation_offset','km')]:metrics[key+'_rms_'+unit]=float(np.sqrt(np.mean((prediction[key]-reference[key])**2)))
    if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite derived field metrics')
    return metrics

def evaluate(prediction_path,evaluation_directory=HERE):
    directory=Path(evaluation_directory);report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((directory/'metrics.json').read_text(encoding='utf8'));boundaries=config['boundaries']
        expected=[f'mesh_{axis}_range_nrmse' for axis in 'xyz']+[name+'_absolute_error' for name in ['latitude_deg','longitude_deg','period_hours']]+['ao_offsets_rms_arcsec','ao_log_gains_rms_dimensionless','occultation_offset_rms_km','complete_weighted_objective']
        if set(boundaries)!=set(expected):raise ValueError('Invalid required metric configuration')
        for metric,boundary in boundaries.items():
            if isinstance(boundary,bool) or not isinstance(boundary,(float,int)) or not np.isfinite(boundary) or boundary<0:raise ValueError('Invalid metric boundary: '+metric)
        report['thresholds']={k:dict(comparison='<=',boundary=v) for k,v in boundaries.items()}
        truth=directory/'ground_truth.npz';reference=_read_npz(truth if truth.exists() or truth.is_symlink() else directory/'baseline_reference.npz')
        prediction=_read_npz(prediction_path);metrics=field_metrics(prediction,reference)
        fields_pass=all(value<=boundaries[key] for key,value in metrics.items())
        # A failed field gate is already decisive. Avoid undefined degenerate
        # forward geometry for otherwise well-formed zero/poor submissions.
        if fields_pass:
            with np.load(directory.parent/'data/raw_data.npz',allow_pickle=False) as archive:raw={k:archive[k] for k in archive.files}
            metrics['complete_weighted_objective']=forward_objective(prediction,raw)
        passed=fields_pass and metrics['complete_weighted_objective']<=boundaries['complete_weighted_objective']
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics)
    except Exception as exc:report.update(status='error',passed=False,metrics={},errors=[str(exc)])
    return report

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:print(f'Cannot write judge report: {exc}',file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[report['status']]

if __name__=='__main__':raise SystemExit(main())
