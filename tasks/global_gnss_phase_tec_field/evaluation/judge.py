"""Independent full-field evaluator. Never imports or executes solver code."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.special import lpmv,gammaln

ROOT=Path(__file__).resolve().parent
SHAPE=(1,25,71,72)

def numeric_boundary(config,key,low,high=None):
    value=config[key]
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
        raise ValueError(f'{key} must be a finite numeric boundary')
    if value<low or (high is not None and value>high):raise ValueError(f'Invalid range for {key}')
    return float(value)

def load_field(path,label):
    if not Path(path).is_file():raise ValueError(f'{label} file does not exist: {path}')
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z.files:raise ValueError(f'{label} missing reconstruction')
        a=z['reconstruction']
        if a.shape!=SHAPE:raise ValueError(f'{label} shape must be {SHAPE}')
        if a.dtype!=np.float64:raise ValueError(f'{label} dtype must be real float64')
        if not np.isfinite(a).all():raise ValueError(f'{label} contains nonfinite values')
        return a.copy()

def harmonics(longitude,colatitude):
    degree=np.array([n for n in range(16) for m in range(-n,n+1)])
    order=np.array([m for n in range(16) for m in range(-n,n+1)])
    am=abs(order)
    norm=np.sqrt((2*degree+1)/(4*np.pi)*np.exp(gammaln(degree-am+1)-gammaln(degree+am+1)))
    result=lpmv(am[None,:],degree[None,:],np.cos(colatitude)[:,None])*norm[None,:]
    neg=order<0;pos=order>0
    result[:,neg]*=np.sqrt(2)*(-1.)**order[neg]*np.sin(am[None,neg]*longitude[:,None])
    result[:,pos]*=np.sqrt(2)*(-1.)**order[pos]*np.cos(am[None,pos]*longitude[:,None])
    return result

def forward_rms(field,raw_path):
    """Fit the declared SH representation using all grid samples, then all rays."""
    with np.load(raw_path,allow_pickle=False) as z:
        lon,colat=np.meshgrid(z['grid_longitude'][0],z['grid_colatitude'][0])
        basis=harmonics(lon.ravel(),colat.ravel())
        c=np.linalg.lstsq(basis,field[0].reshape(25,-1).T,rcond=None)[0].T
        d={k:z[k][0] for k in ['time','time_ref','mlt1','mcolat1','el','mlt1_ref','mcolat1_ref','el_ref','rhs']}
    prediction=np.zeros(len(d['rhs']))
    for start in range(0,len(prediction),4096):
        end=min(start+4096,len(prediction));ix=slice(start,end)
        for suffix,sign in [('',1.),('_ref',-1.)]:
            u=d['time'+suffix][ix]/3600;j=np.floor(u).astype(int);w=u-j
            h=harmonics(d['mlt1'+suffix][ix],d['mcolat1'+suffix][ix])
            slant=1/np.sqrt(1-(6371000*np.cos(d['el'+suffix][ix])/6821000)**2)
            prediction[ix]+=sign*slant*np.einsum('ij,ij->i',h,(1-w[:,None])*c[j]+w[:,None]*c[j+1])
    a=np.sin(d['el'])**2;b=np.sin(d['el_ref'])**2;weights=a*b/(a+b)
    return float(np.sqrt(np.dot(weights,(prediction-d['rhs'])**2)/weights.sum()))

def evaluate(prediction,asset_dir=ROOT):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        root=Path(asset_dir)
        config=json.loads((root/'metrics.json').read_text(encoding='utf8'))
        ncc_limit=numeric_boundary(config,'ncc_boundary',-1,1)
        error_limit=numeric_boundary(config,'nrmse_boundary',0)
        residual_limit=numeric_boundary(config,'forward_weighted_rms_boundary',0)
        report['thresholds']={'ncc_min':{'comparison':'>=','boundary':ncc_limit},
            'nrmse_max':{'comparison':'<=','boundary':error_limit},
            'forward_weighted_rms_TECu':{'comparison':'<=','boundary':residual_limit}}
        # Actual truth takes precedence, including when it is malformed.
        truth=root/'ground_truth.npz'
        reference=load_field(truth if truth.exists() else root/'baseline_reference.npz','Reference')
        field=load_field(prediction,'Prediction')
        r=reference.reshape(25,-1);x=field.reshape(25,-1);ranges=np.ptp(r,axis=1)
        rnorm=np.linalg.norm(r,axis=1)
        if np.any(ranges<=0) or np.any(rnorm<=0):raise ValueError('Degenerate reference map')
        denominator=np.linalg.norm(x,axis=1)*rnorm
        cosine=np.divide(np.sum(x*r,axis=1),denominator,out=np.zeros(25),where=denominator>0)
        errors=np.sqrt(np.mean((x-r)**2,axis=1))/ranges
        if not np.isfinite(cosine).all() or not np.isfinite(errors).all():
            raise ValueError('Nonfinite metric arithmetic, including finite-value overflow')
        report['metrics']={'ncc_min':float(cosine.min()),'nrmse_max':float(errors.max())}
        good=bool(np.all(cosine>=ncc_limit) and np.all(errors<=error_limit))
        # A failed full-field gate needs no costly full-ray follow-up.
        if good:
            residual=forward_rms(field,root.parent/'data/raw_data.npz')
            if not np.isfinite(residual):raise ValueError('Nonfinite forward metric')
            report['metrics']['forward_weighted_rms_TECu']=residual
            good=bool(residual<=residual_limit)
        report.update(status='pass' if good else 'fail',passed=good)
        return report,0 if good else 1
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},errors=[str(exc)])
        return report,2

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True)
    args=p.parse_args();report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print(f'Report write failed: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
