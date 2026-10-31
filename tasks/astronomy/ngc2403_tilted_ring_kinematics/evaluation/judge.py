"""Standalone profile scoring plus explicit observation-consistency physics.

The forward formulas are from the same independently validated published model;
they are deliberately embedded here to avoid importing reference solver code.
No inverse method, hidden fitted gain, output cache or submitted code is used.
"""
from pathlib import Path
import argparse,json,sys,math
from functools import lru_cache
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

EVALUATION=Path(__file__).resolve().parent
KEYS=('rotation','dispersion','position_angle')
BOUNDARIES=('rotation_range_nrmse_boundary','dispersion_range_nrmse_boundary','position_angle_wrapped_rms_deg_boundary','full_cube_rms_Jy_beam_boundary')

SPATIAL_STEP_ARCMIN=np.array([-.266666673126,.266666673126],dtype=float)


CHANNEL_STEP_M_S=-5152.860000000001


VELOCITY_FIRST_M_S=289610.96899999998


def beam_kernel(major_arcsec=59.3287870284, minor_arcsec=58.6053617292,
                angle_deg=-79.446372986, cutoff=1e-4):
    """Native Gaussian kernel, elliptical cutoff, source rounded extent.

    FWHM beam angles are sky PA; the array x-axis rotation is PA+90 degrees.
    Spatial pixel signs enter the sky model; kernel spacings are absolute.
    """
    # Convpars stores atan2(sin(2PA),cos(2PA)) in template T=float32
    # before dividing by two. Preserve that discretization, not ideal PA.
    native_twice=np.float32(math.atan2(math.sin(2*np.deg2rad(angle_deg)),math.cos(2*np.deg2rad(angle_deg))))
    phi=float(native_twice)/2+np.pi/2
    cs,sn=math.cos(phi),math.sin(phi)
    extend=math.sqrt(-math.log(cutoff)/math.log(2))
    rx=.5*major_arcsec*extend;ry=.5*minor_arcsec*extend
    px,py=abs(SPATIAL_STEP_ARCMIN)*60
    hx=int(nearest_integer(max(abs(rx*cs),abs(ry*sn))/px))
    hy=int(nearest_integer(max(abs(rx*sn),abs(ry*cs))/py))
    yy,xx=np.indices((2*hy+1,2*hx+1));x=(xx-hx)*px;y=(yy-hy)*py
    xr=x*cs+y*sn;yr=-x*sn+y*cs
    values=np.exp(-4*np.log(2)*((xr/major_arcsec)**2+(yr/minor_arcsec)**2))
    values=np.where(values>=cutoff,values,0)
    total=np.add.accumulate(values.ravel())[-1]
    return values/total


def internal_rings(radius_edges_arcsec, parameters, density=1.0):
    """Float32 radial quadrature, selected fixed geometry and native bounding box.

    parameters rows are [rotation km/s,dispersion km/s,position angle deg].
    A single row defines one fitted annulus; multiple rows interpolate a disk.
    Returns internal rings, metadata and [radius,PA,inc,x0,y0] edge geometry.
    """
    radius=np.asarray(radius_edges_arcsec,dtype=np.float32)
    p=np.asarray(parameters,dtype=np.float32)
    if p.ndim==1:p=np.repeat(p[None],radius.size,axis=0)
    if p.shape!=(radius.size,3):raise ValueError('One parameter row per radial knot is required')
    # Each assignment here is float32 in the original ring container.
    r=(radius.astype(float)/60).astype(np.float32)
    width=.75*min(abs(SPATIAL_STEP_ARCMIN))
    start=r[0]+width/2 if r[0]!=0 else width/2
    sample=np.float32(start)
    values=np.column_stack((
        (p[:,0]*np.float32(1000)).astype(np.float32),
        (p[:,1]*np.float32(1000)).astype(np.float32),
        np.full(radius.size,np.float32(np.deg2rad(60))),
        (p[:,2].astype(float)*np.pi/180).astype(np.float32),
        np.broadcast_to(np.asarray(density,dtype=np.float32),(radius.size,)),
        np.full(radius.size,np.float32(10/60)),np.full(radius.size,77,dtype=np.float32),
        np.full(radius.size,77,dtype=np.float32),np.full(radius.size,np.float32(132.8)*np.float32(1000)))).astype(np.float32)
    output=[]
    for i in range(1,radius.size):
        slope=((values[i]-values[i-1])/np.float32(r[i]-r[i-1])).astype(np.float32)
        while sample<r[i]:
            delta=np.float32(sample-r[i-1])
            interpolated=(values[i-1]+slope*delta).astype(np.float32)
            output.append(np.r_[float(sample),interpolated.astype(float),62.])
            sample=np.float32(float(sample)+width)
    rings=np.asarray(output,dtype=float)
    # A freshly addRings-constructed fitted annulus retains radsep=0.
    # Its outer edge already includes the 15-arcsec half width.
    extent=int(math.ceil((float(radius[-1])+3*10)/(abs(SPATIAL_STEP_ARCMIN[0])*60)))
    bx=max(0,77-extent);by=bx;ex=min(150,77+extent+1);ey=ex
    px=abs(SPATIAL_STEP_ARCMIN[0]);py=abs(SPATIAL_STEP_ARCMIN[1]);instrument=float(np.float32(1/(2*np.sqrt(2*np.log(2)))))*abs(CHANNEL_STEP_M_S)
    meta=np.array([len(rings),ex-bx,ey-by,62,bx,by,ex,ey,*SPATIAL_STEP_ARCMIN,1,VELOCITY_FIRST_M_S,CHANNEL_STEP_M_S,abs(CHANNEL_STEP_M_S),instrument,px*py,width])
    edges=np.column_stack((radius,p[:,2],np.full(radius.size,60),np.full(radius.size,77),np.full(radius.size,77)))
    return rings,meta,edges


def zero_moment(cube,mask):
    """Masked integrated intensity, Jy/beam km/s, original float32 summation."""
    total=np.add.accumulate((np.asarray(cube,dtype=np.float32)*mask).astype(np.float32),axis=0,dtype=np.float32)[-1]
    value=(total.astype(float)*abs(CHANNEL_STEP_M_S)/1000).astype(np.float32)
    return np.where(total!=0,value,np.float32(np.nan))


def azimuthal_profile(moment0,parameters):
    """Author full-ring 2x2 subpixel flux estimate, including blank accounting.

    Columns are [integrated sum,number of valid subpixels,blank subpixels,
    sky-plane surface brightness,face-on surface brightness]. The surface
    brightness units are Jy/beam km/s/arcsec². This estimates nuisance flux
    from observations; it is not a gas-density truth or an added calibration.
    """
    image=np.asarray(moment0,dtype=np.float32)
    p=np.asarray(parameters,dtype=np.float32)
    yy,xx=np.indices(image.shape)
    # Author Dx/Dy and 2x2 subdivision are stored as float32 (16 and 8).
    spacing=np.float32(abs(SPATIAL_STEP_ARCMIN[0])*60)
    xcenter=(spacing*(xx-77)).astype(np.float32)
    ycenter=(spacing*(yy-77)).astype(np.float32)
    # x-major, y, subpixel-x, subpixel-y is the original accumulation order.
    offsets=np.array([[-spacing/4,-spacing/4],[-spacing/4,spacing/4],
                      [spacing/4,-spacing/4],[spacing/4,spacing/4]],dtype=np.float32)
    x=(xcenter.T.ravel()[:,None]+offsets[None,:,0]).ravel()
    y=(ycenter.T.ravel()[:,None]+offsets[None,:,1]).ravel()
    values=np.repeat(image.T.ravel(),4)
    valid=np.isfinite(values)
    radius=np.arange(15,1216,30,dtype=np.float32)
    output=[]
    for r,angle in zip(radius,p[:,2]):
        phi=-(float(angle)+90)*np.pi/180
        cosphi,sinphi=np.cos(phi),np.sin(phi)
        xp=(x.astype(float)*cosphi-y.astype(float)*sinphi).astype(np.float32)
        yp=((x.astype(float)*sinphi+y.astype(float)*cosphi)/np.cos(np.pi/3)).astype(np.float32)
        squared=xp*xp+yp*yp
        inner=np.float32(max(float(r)-15,0));outer=np.float32(float(r)+15)
        inside=(squared>=inner*inner)&(squared<outer*outer)
        n=int(np.count_nonzero(inside&valid));blanks=int(np.count_nonzero(inside&~valid))
        total=float(np.add.accumulate(values[inside&valid].astype(float))[-1])/4 if n else 0.
        # Native Num/4 and area are float expressions before double assignment.
        area=np.float32(np.float32(n)/np.float32(4))*np.float32(spacing*spacing)
        surface=np.float32(total/float(area)) if area else np.float32(0)
        output.append([total,n,blanks,float(surface),float(surface)*np.cos(np.pi/3)])
    return np.asarray(output)


def full_model(cube,mask,parameters,seed=-1):
    """Complete original post-fit disk emission, with observed annular flux.

    The lowest positive face-on profile selects the author's power-of-ten
    scaling to keep Monte-Carlo fluxes away from numerical underflow.
    No missing sky/velocity voxels are excluded from the returned cube.
    """
    profile=azimuthal_profile(zero_moment(cube,mask),parameters)
    positive=profile[:,-1][profile[:,-1]>0]
    if not positive.size:raise ValueError('No positive observation-derived radial flux')
    minimum=float(positive.min());scale=np.float32(1)
    while minimum<.1:minimum*=10;scale=np.float32(scale*10)
    while minimum>10:minimum/=10;scale=np.float32(scale/10)
    density=(profile[:,-1]*float(scale)).astype(np.float32)
    radius=np.arange(15,1216,30,dtype=np.float32);radius[0]=0;radius[-1]+=15
    rings,metadata,_=internal_rings(radius,parameters,density)
    # Output uses the complete native observation box, regardless of geometry.
    metadata=metadata.copy();metadata[1:8]=[150,150,62,0,0,150,150]
    model=beam_convolve(cloud_cube(rings,metadata,seed=seed),beam_kernel())
    return (model/scale).astype(np.float32),profile


def beam_convolve(cube, kernel, scale=1.0):
    """Zero-padded, linear SAME beam convolution; source positive-tail cutoff."""
    result=fftconvolve(np.asarray(cube,dtype=np.float64),np.asarray(kernel)[None],mode='same',axes=(-2,-1))
    return np.where(result<1e-12,0,result*scale).astype(np.float32)


class CloudRandom:
    """General MT19937 plus explicit libstdc++ float canonical/polar sampling.

    This is an integration convention, not a claim that a seed establishes
    scientific quality. Complete independent-seed runs define the field gates.
    A uint32 pool and vectorized pair rejection leave the mathematical draws
    unchanged. No NumPy normal-distribution implementation is called.
    """
    def __init__(self, seed=-1):
        self.generator=np.random.RandomState(seed&0xffffffff)
        self.pool=np.empty(0,dtype=np.uint32);self.cursor=0
        self.saved=None

    def _ensure(self,n):
        if self.cursor+n>self.pool.size:
            tail=self.pool[self.cursor:]
            self.pool=np.r_[tail,self.generator.randint(0,2**32,size=max(65536,n),dtype=np.uint32)]
            self.cursor=0

    def canonical(self,n=1):
        self._ensure(n)
        raw=self.pool[self.cursor:self.cursor+n].astype(np.float32)
        self.cursor+=n
        canonical=np.minimum(raw*np.float32(2**-32),np.nextafter(np.float32(1),np.float32(0)))
        return canonical

    def uniform(self,n=1):
        return self.canonical(n)*np.float32(2)-np.float32(1)

    def normal(self,n):
        parts=[];left=n
        if self.saved is not None:
            parts.append(np.array([self.saved],dtype=np.float32));left-=1;self.saved=None
        while left:
            pairs=max(8,int(math.ceil(left/.75)))
            values=self.uniform(2*pairs).reshape(-1,2)
            radius=(values*values).sum(axis=1,dtype=np.float32)
            indices=np.flatnonzero((radius<=1)&(radius!=0))
            need=(left+1)//2
            chosen=indices[:need]
            if not chosen.size:
                continue
            # Undo only unused look-ahead words; preserve accepted/rejected pairs.
            # _ensure may have moved the pool, so reconstruct from current cursor.
            self.cursor-=2*(pairs-int(chosen[-1])-1)
            r=radius[chosen]
            mult=np.sqrt((-np.float32(2)*np.log(r))/r,dtype=np.float32)
            generated=(values[chosen][:,::-1]*mult[:,None]).ravel()
            take=min(left,generated.size)
            parts.append(generated[:take]);left-=take
            if generated.size>take:
                self.saved=generated[-1]
        return np.concatenate(parts)


def nearest_integer(value):
    """C++ lround: halfway cases go away from zero."""
    value=np.asarray(value)
    return np.where(value>=0,np.floor(value+.5),np.ceil(value-.5)).astype(np.int64)


def cloud_cube(internal_rings, metadata, seed=-1, cdens=10.0, quadrature=None):
    """Selected optically-thin H I cloud integral on a native velocity grid.

    Internal columns: radius arcmin,Vrot/disp m/s,inc/PA rad,density,z0 arcmin,
    center x/y pixel,Vsys m/s,velocity samples per cloud. Actual off-box clouds
    consume their spatial draws and skip all velocity draws, as in the source.
    Optional quadrature is an explicit deterministic integration input with
    columns ring index,sky-profile index,cos(azimuth),62 standard LOS deviates;
    frozen original quadrature separates operator parity from stochastic tests.
    """
    nr,nx,ny,nz,bx,by,ex,ey=map(int,metadata[:8])
    dx,dy,crpix,crval,dv,width,instrument,pixarea,dr=metadata[8:]
    out=np.zeros((nz,ny,nx),dtype=np.float32)
    # Geometry and random quadrature do not depend on Vrot/dispersion.
    # Reusing exactly those computed draws is the same source common-random
    # integration, and does not cache fitted fields, objective values or cubes.
    geometry=np.asarray(internal_rings)[:,[0,3,4,5,6,7,8,9,10]]
    if quadrature is None:
        samples=_cloud_samples(tuple(geometry.ravel()),tuple(np.asarray(metadata)),seed,float(cdens))
    else:
        samples=[]
        quadrature=np.asarray(quadrature,dtype=float)
        for ir,row in enumerate(internal_rings):
            radius,_,_,_,_,density,_,_,_,_,nv=row
            nv=int(nv);area=2*np.pi*radius*dr/pixarea
            clouds=int(nearest_integer(cdens*density*area))
            if nearest_integer(clouds/area)<10:clouds=int(nearest_integer(10*area))
            increment=np.float32(np.float32(density*area*pixarea/(clouds*nv))*np.float32(3600/width*1000))
            selected=quadrature[quadrature[:,0]==ir]
            if selected.shape[1]!=3+nv:raise ValueError('Quadrature velocity-sample count mismatch')
            samples.append((selected[:,1].astype(np.int64),selected[:,2],selected[:,3:].astype(np.float32),increment))
    for row,(profiles,cosaz,deviates,increment) in zip(internal_rings,samples):
        radius,rotation,dispersion,inc,pa,density,z0,x0,y0,systemic,nv=row
        if not profiles.size:continue
        sigma=math.sqrt(dispersion*dispersion+instrument*instrument)
        velocity=systemic+rotation*cosaz[:,None]*math.sin(inc)+deviates.astype(float)*sigma
        channel=nearest_integer((velocity-crval)/dv+crpix-1)
        index=channel*(nx*ny)+profiles[:,None]
        use=(channel>=0)&(channel<nz)
        np.add.at(out.ravel(),index[use],increment)
    return out


@lru_cache(maxsize=16)
def _cloud_samples(geometry_key,metadata_key,seed,cdens):
    """Memoize only source random quadrature at identical physical geometry."""
    metadata=np.asarray(metadata_key)
    nr,nx,ny,nz,bx,by,ex,ey=map(int,metadata[:8])
    dx,dy,crpix,crval,dv,width,instrument,pixarea,dr=metadata[8:]
    random=CloudRandom(seed);samples=[]
    for row in np.asarray(geometry_key).reshape(-1,9):
        radius,inc,pa,density,z0,x0,y0,systemic,nv=row
        if density==0:
            samples.append((np.empty(0,dtype=np.int64),np.empty(0),np.empty((0,int(nv)),dtype=np.float32),np.float32(0)))
            continue
        nv=int(nv)
        area=2*np.pi*radius*dr/pixarea
        clouds=int(nearest_integer(cdens*density*area))
        if nearest_integer(clouds/area)<10:
            clouds=int(nearest_integer(10*area))
        flux=np.float32(density*area*pixarea/(clouds*nv))
        cd2i=np.float32(3600/width*1000)
        increment=np.float32(np.float32(1)*flux*cd2i)
        sini=math.sin(inc);cosi=math.cos(inc);sinpa=math.sin(pa);cospa=math.cos(pa)
        profiles=[];cosaz=[];deviates=[]
        for _ in range(clouds):
            u=random.uniform(3).astype(float)
            R=math.sqrt((radius-.5*dr)**2+2*dr*radius*abs(u[0]))
            az=2*np.pi*abs(u[1]);height=math.atanh(u[2])*z0
            x=R*math.cos(az);y=R*math.sin(az)*cosi-height*sini
            gx=int(nearest_integer(x0+(x*sinpa-y*cospa)/dx))
            gy=int(nearest_integer(y0+(x*cospa+y*sinpa)/dy))
            if not (bx<=gx<ex and by<=gy<ey):continue
            profiles.append((gy-by)*nx+gx-bx);cosaz.append(math.cos(az));deviates.append(random.normal(nv))
        samples.append((np.asarray(profiles,dtype=np.int64),np.asarray(cosaz),np.asarray(deviates,dtype=np.float32).reshape(-1,nv),increment))
    return tuple(samples)


VELOCITY_START_KM_S=289.610969


VELOCITY_STEP_KM_S=-5.152860000000001


PIXEL_ARCSEC=16.00000038756


def _components(array):
    """Chebyshev sky-adjacency 1 and spectral separation <=2 channels."""
    labels,count=ndimage.label(array,structure=np.ones((3,3,3),dtype=bool))
    parent=np.arange(count+1,dtype=np.int32)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for dy in [-1,0,1]:
        ya=slice(max(0,dy),min(array.shape[1],array.shape[1]+dy))
        yb=slice(max(0,-dy),min(array.shape[1],array.shape[1]-dy))
        for dx in [-1,0,1]:
            xa=slice(max(0,dx),min(array.shape[2],array.shape[2]+dx))
            xb=slice(max(0,-dx),min(array.shape[2],array.shape[2]-dx))
            a=labels[:-2,ya,xa];b=labels[2:,yb,xb]
            valid=(a!=0)&(b!=0)&(a!=b)
            pairs=np.unique(np.stack((a[valid],b[valid]),axis=1),axis=0)
            for i,j in pairs:
                ri,rj=root(i),root(j)
                if ri!=rj:parent[rj]=ri
    lookup=np.array([root(i) for i in range(count+1)])
    return lookup[labels]


def search_mask(cube):
    """Robust 4-sigma source search grown to 2.5-sigma; largest object.

    Blank zero observations are excluded from the median/MAD statistics.
    Source minimum support is ceil(Gaussian beam area)=16 sky pixels and
    at least two channels / 32 voxels, before and after source merging.
    """
    cube=np.asarray(cube,dtype=np.float32)
    valid=np.isfinite(cube)&(cube!=0)
    values=cube[valid]
    middle=np.float32(np.median(values))
    mad=np.float32(np.median(np.abs(values-middle)))
    spread=np.float32(mad/np.float32(.6744888))
    threshold=np.float32(middle+np.float32(4)*spread)
    growth=np.float32(middle+np.float32(2.5)*spread)
    high=_components(cube>threshold)
    high_ids=np.unique(high[high!=0])
    selected=[]
    for identifier in high_ids:
        z,y,x=np.where(high==identifier)
        if z.size<32 or np.unique(y*cube.shape[2]+x).size<16:continue
        channels=np.unique(z)
        if not np.any(np.diff(channels)==1):continue
        selected.append(identifier)
    seed=np.isin(high,selected)&(high!=0)
    low=_components(cube>growth)
    candidates=np.unique(low[seed]);candidates=candidates[candidates!=0]
    if not candidates.size:
        raise ValueError('No emission source satisfies the declared SEARCH support')
    sizes=np.bincount(low.ravel())
    identifier=candidates[np.argmax(sizes[candidates])]
    return low==identifier,dict(median=float(middle),mad=float(mad),spread=float(spread),threshold=float(threshold),growth_threshold=float(growth))


def _arrays(path):
    path=Path(path)
    if not path.is_file():raise ValueError(f'Missing NPZ: {path}')
    arrays={}
    with np.load(path,allow_pickle=False) as archive:
        for key in KEYS:
            if key not in archive:raise ValueError(f'Missing required key: {key}')
            value=archive[key]
            if value.shape!=(1,41):raise ValueError(f'{key}: expected shape(1,41)')
            if value.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError(f'{key}: expected real float32/float64')
            if not np.isfinite(value).all():raise ValueError(f'{key}: nonfinite values')
            if key!='position_angle' and np.any(value<0):raise ValueError(f'{key}: negative physical magnitude')
            arrays[key]=value.astype(float)
    return arrays


def load_reference(directory):
    directory=Path(directory)
    truth=directory/'ground_truth.npz'
    # Existing invalid truth is an evaluator error, never a fallback signal.
    chosen=truth if truth.exists() else directory/'baseline_reference.npz'
    return _arrays(chosen)


def load_configuration(path):
    with Path(path).open(encoding='utf-8') as stream:config=json.load(stream)
    bounds={}
    for key in BOUNDARIES:
        value=config.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError(f'Invalid finite nonnegative numeric boundary: {key}')
        bounds[key]=float(value)
    return bounds


def profile_metrics(estimate,reference):
    result={}
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        for key in KEYS[:2]:
            span=float(np.ptp(reference[key]))
            if not math.isfinite(span) or span<=0:raise ValueError(f'Degenerate {key} reference dynamic range')
            error=float(np.sqrt(np.mean((estimate[key]-reference[key])**2))/span)
            result[key+'_range_nrmse']=error
        delta=(estimate['position_angle']-reference['position_angle']+180)%360-180
        result['position_angle_wrapped_rms_deg']=float(np.sqrt(np.mean(delta**2)))
    if not all(math.isfinite(v) for v in result.values()):raise ValueError('Profile metric overflow or undefined result')
    return result


def evaluate(prediction,reference_dir=None,data_path=None):
    directory=EVALUATION if reference_dir is None else Path(reference_dir)
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        bounds=load_configuration(directory/'metrics.json')
        report['thresholds']={key.removesuffix('_boundary'):dict(comparison='<=',boundary=value) for key,value in bounds.items()}
        reference=load_reference(directory)
        estimate=_arrays(prediction)
        measured=profile_metrics(estimate,reference)
        # Early rejection preserves all three profile scores and cannot pass
        # without evaluating the full additional physical criterion.
        profile_pass=all(value<=bounds[key+'_boundary'] for key,value in measured.items())
        report['metrics']=measured
        if not profile_pass:
            report.update(status='fail',passed=False,observation_consistency_evaluated=False)
            return report,1
        path=directory.parent/'data/raw_data.npz' if data_path is None else Path(data_path)
        with np.load(path,allow_pickle=False) as archive:observation=archive['spectral_cube']
        if observation.shape!=(1,62,150,150) or observation.dtype!=np.float32 or not np.isfinite(observation).all():
            raise ValueError('Invalid public spectral_cube asset')
        fields=np.column_stack([estimate[key][0] for key in KEYS]);fields[:,2]%=360
        valid,_=search_mask(observation[0])
        modeled,_=full_model(observation[0],valid,fields)
        with np.errstate(over='ignore',invalid='ignore'):
            rms=float(np.sqrt(np.mean((modeled.astype(float)-observation[0])**2)))
        if not math.isfinite(rms):raise ValueError('Forward residual overflow or undefined result')
        measured['full_cube_rms_Jy_beam']=rms
        passed=rms<=bounds['full_cube_rms_Jy_beam_boundary']
        report.update(status='pass' if passed else 'fail',passed=bool(passed),observation_consistency_evaluated=True)
        return report,0 if passed else 1
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},errors=[str(exc)])
        return report,2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:
        print(f'Cannot write Judge report: {exc}',file=sys.stderr)
        return 2
    return code


if __name__=='__main__':raise SystemExit(main())
