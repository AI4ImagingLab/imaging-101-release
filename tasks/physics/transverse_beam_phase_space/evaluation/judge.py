"""Independent finite-volume phase-space evaluation; no solver imports."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

BASE_NAMES=['normalization_abs_error','negative_probability_max','whitening_max_abs_error','relative_entropy_KL']
NAMES=BASE_NAMES+[f'profile_{i:02d}_relative_l2' for i in range(24)]
COORDS=np.linspace(-85.,85.,100)

def configuration(path):
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict):raise ValueError('Metric configuration must be a JSON object')
    if {k for k in value if k.endswith('_boundary')}!={k+'_boundary' for k in NAMES}:raise ValueError('Metric boundary set differs from declared28 metrics')
    out={}
    for key in NAMES:
        v=value[key+'_boundary']
        if type(v) not in (int,float) or not math.isfinite(v) or v<0:raise ValueError('Metric boundaries must be finite nonnegative JSON numbers')
        out[key]={'comparison':'<=','boundary':v}
    if 'comparisons' in value and value['comparisons']!={k:'<=' for k in NAMES}:raise ValueError('Invalid metric comparisons')
    return out

def observations(path):
    with np.load(path,allow_pickle=False) as z:
        required={'raw_coordinates','raw_signal','sample_mask','profile_mean','profile_sigma','transfer_matrices'}
        if set(z.files)!=required:raise ValueError('Invalid evaluator observation keys')
        a={k:z[k] for k in required}
    if a['raw_coordinates'].shape!=(1,8,3,90) or a['raw_signal'].shape!=(1,8,3,90) or a['sample_mask'].shape!=(1,8,3,90) or a['sample_mask'].dtype!=np.bool_:raise ValueError('Invalid original profile shapes/mask')
    if a['profile_mean'].shape!=(1,8,3) or a['profile_sigma'].shape!=(1,8,3) or a['transfer_matrices'].shape!=(1,8,4,4):raise ValueError('Invalid evaluator moment/transport shapes')
    if any(a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in required-{'sample_mask'}):raise ValueError('Invalid original precision or finite values')
    if (a['profile_sigma']<=0).any():raise ValueError('Nonpositive observed RMS width')
    profiles=[]
    for i in range(8):
        group=[]
        for j in range(3):
            keep=a['sample_mask'][0,i,j]
            if keep.sum()<3 or not np.array_equal(keep,np.arange(90)<keep.sum()):raise ValueError('Invalid acquisition sample padding')
            x=a['raw_coordinates'][0,i,j,keep]-a['profile_mean'][0,i,j]
            y=a['raw_signal'][0,i,j,keep].copy()
            if j==1:x=-x[::-1];y=y[::-1]
            cutoff=.01*y.max()
            if y.min()<0:y-=y.min()
            y[y<cutoff]=0
            selected=np.r_[True,abs(np.diff(y))/y.max()<=.25]
            interpolated=interp1d(x[selected],y[selected],kind='linear',bounds_error=False,fill_value=0.)(COORDS)
            interpolated=gaussian_filter(interpolated,1.)
            interpolated-=interpolated.min()
            group.append(interpolated/interpolated.sum()/(COORDS[1]-COORDS[0]))
        profiles.append(group)
    matrices=a['transfer_matrices'][0]
    moments=a['profile_sigma'][0]**2
    # Independent symmetric-matrix linear design rather than three solver blocks.
    pairs=[(i,j) for i in range(4) for j in range(i,4)]
    design=[];targets=[]
    for M,(xx,yy,uu) in zip(matrices,moments):
        for left,right,value in [(M[0],M[0],xx),(M[2],M[2],yy),(M[0],M[2],.5*(xx+yy)-uu)]:
            design.append([left[i]*right[j]+(left[j]*right[i] if i!=j else 0.) for i,j in pairs])
            targets.append(value)
    vector=np.linalg.lstsq(design,targets,rcond=None)[0]
    covariance=np.zeros((4,4))
    for (i,j),value in zip(pairs,vector):covariance[i,j]=value;covariance[j,i]=value
    if np.linalg.eigvalsh(covariance).min()<=0:raise ValueError('Evaluator moment inversion is not positive definite')
    return np.asarray(profiles),matrices,covariance

def uniform_voxel_projection(mass,centers,linear,coords,subdivisions=16):
    # Piecewise-constant 4D voxel density under a linear scalar instrument map.
    dx=coords[1]-coords[0];step=dx/subdivisions
    edge=coords[0]-dx/2
    projected=centers@linear
    widths=np.abs(linear)*(8/35)
    lo=int(np.floor((projected.min()-widths.sum()/2-edge)/step))-4
    hi=int(np.ceil((projected.max()+widths.sum()/2-edge)/step))+4
    lo=min(lo,0);hi=max(hi,len(coords)*subdivisions+subdivisions//2)
    index=(projected-edge)/step-lo-.5
    left=np.floor(index).astype(int);fraction=index-left
    count=hi-lo
    deposited=np.bincount(left,weights=mass*(1-fraction),minlength=count)
    deposited+=np.bincount(left+1,weights=mass*fraction,minlength=count)
    footprint=np.ones(1)
    for width in widths:
        if width<1e-14:continue
        radius=int(np.ceil(width/2/step+.5))
        c=np.arange(-radius,radius+1)*step
        p=np.maximum(0,np.minimum(c+step/2,width/2)-np.maximum(c-step/2,-width/2))/width
        footprint=np.convolve(footprint,p)
    projected_mass=np.convolve(deposited,footprint,mode='same')
    result=projected_mass[-lo:-lo+len(coords)*subdivisions].reshape(len(coords),subdivisions).sum(axis=1)
    # Preserve the source's explicit terminal histogram edge at last_center+dx.
    # Its last bin is1.5dx; source converts probability to density before its
    # final equal-center quadrature normalization.
    result[-1]+=projected_mass[-lo+len(coords)*subdivisions:-lo+len(coords)*subdivisions+subdivisions//2].sum()
    result[-1]/=1.5
    return result/result.sum()/dx

def score(mass,N,measured,matrices,S,whitening_boundary=1e-5):
    metrics={'normalization_abs_error':abs(float(mass.sum())-1),
             'negative_probability_max':max(0.,-float(mass.min())),
             'whitening_max_abs_error':float(np.max(abs(N@S@N.T-np.eye(4))))}
    if not all(math.isfinite(v) for v in metrics.values()):raise ValueError('Prediction overflows physical constraints')
    if mass.sum()<=0 or mass.min()<-1e-12 or np.linalg.matrix_rank(N)!=4 or metrics['whitening_max_abs_error']>whitening_boundary:return metrics
    p=np.maximum(mass.ravel(),0);p/=p.sum()
    axis=(np.linspace(-4,4,36)[:-1]+np.linspace(-4,4,36)[1:])/2
    centers=np.stack(np.meshgrid(*[axis]*4,indexing='ij'),axis=-1).reshape(-1,4)
    prior=np.exp(-.5*np.sum(centers**2,axis=1));prior/=prior.sum()
    valid=p>0
    metrics['relative_entropy_KL']=float(np.sum(p[valid]*np.log(p[valid]/prior[valid])))
    V=np.linalg.inv(N)
    directions=np.array([[1.,0,0,0],[0,0,1.,0],[1.,0,-1.,0]])/np.array([[1.],[1.],[np.sqrt(2)]])
    for i in range(8):
        for j in range(3):
            linear=directions[j]@matrices[i]@V
            projected=uniform_voxel_projection(p,centers,linear,COORDS,32)
            metric=float(np.linalg.norm(projected-measured[i,j])/np.linalg.norm(measured[i,j]))
            metrics[f'profile_{3*i+j:02d}_relative_l2']=metric
    if not all(math.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite physical evaluation metric')
    return metrics

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    code=2;root=Path(__file__).resolve().parent
    try:
        report['thresholds']=configuration(root/'metrics.json')
        measured,matrices,S=observations(root.parent/'data/raw_data.npz')
        with np.load(args.prediction,allow_pickle=False) as z:
            if set(z.files)!={'reconstruction','normalization'}:raise ValueError('Expected exactly reconstruction and normalization')
            mass,N=z['reconstruction'],z['normalization']
        if mass.shape!=(1,35,35,35,35) or N.shape!=(1,4,4):raise ValueError('Wrong batch-first output shapes')
        for array in [mass,N]:
            if array.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Output must be real float32/float64')
            if not np.isfinite(array).all():raise ValueError('Nonfinite prediction values')
        report['metrics']=score(mass[0].astype(np.float64),N[0].astype(np.float64),measured,matrices,S,whitening_boundary=report['thresholds']['whitening_max_abs_error']['boundary'])
        passed=set(report['metrics'])==set(NAMES) and all(report['metrics'][k]<=report['thresholds'][k]['boundary'] for k in report['metrics'])
        report.update(status='pass' if passed else 'fail',passed=passed)
        if set(report['metrics'])!=set(NAMES):report['errors'].append('Prediction outside nonnegative normalized/invertible whitening physical domain')
        code=0 if passed else 1
    except Exception as exc:report.update(status='error',passed=False,metrics={},errors=[f'{type(exc).__name__}: {exc}'])
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:
        print(f'Cannot write evaluation report: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':
    if hasattr(sys.stderr,'reconfigure'):sys.stderr.reconfigure(encoding='utf-8',errors='backslashreplace')
    raise SystemExit(main())
