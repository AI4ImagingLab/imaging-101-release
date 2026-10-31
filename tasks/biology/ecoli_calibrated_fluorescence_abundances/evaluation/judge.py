"""Standalone evaluator; no reference solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def regularized_score(x,y,m):
    """Scale-profiled convex KL/nuclear/group target, digital intensities."""
    observed=y[0,1:-1,1:-1].reshape(-1,24).T.astype(np.float64)/255.
    observed/=np.sqrt(np.mean(observed**2))
    abundance=x[0,1:-1,1:-1].reshape(-1,13).T.astype(np.float64)
    prediction=m@abundance
    penalty=.1*(np.linalg.svd(abundance,compute_uv=False).sum()+np.linalg.norm(abundance,axis=1).sum())
    denominator=float(prediction.sum()+penalty)
    if denominator<=0:return 1e300
    scale=float(observed.sum()/denominator)
    prediction=prediction*scale
    if np.any((prediction<=0)&(observed>0)):return 1e300
    positive=observed>0
    divergence=float(np.sum(prediction-observed)+np.sum(observed[positive]*np.log(observed[positive]/prediction[positive])))
    return float((divergence+scale*penalty)/observed.size)
def validate(a):
    if a.shape!=(1,30,30,13):raise ValueError('reconstruction shape must be (1,30,30,13)')
    if a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('reconstruction dtype must be real float32 or float64')
    if not np.isfinite(a).all():raise ValueError('reconstruction contains NaN or Inf')

def metrics(x,r,y,m):
    a,b=x.ravel().astype(float),r.ravel().astype(float)
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    ncc=float(a@b/denom) if denom else 0.
    dr=float(np.ptp(b))
    if dr<=0:raise ValueError('constant reference is an invalid evaluator configuration')
    rmse=float(np.sqrt(np.mean((a-b)**2))/dr)
    p=(x[0,1:-1,1:-1]@m.T).ravel();obs=y[0,1:-1,1:-1].ravel().astype(float)
    alpha=max(0,float(p@obs/(p@p))) if p@p else 0.
    residual=float(np.linalg.norm(alpha*p-obs)/np.linalg.norm(obs))
    return {'ncc':ncc,'nrmse':rmse,'observation_relative_error':residual}

def evaluate(prediction,assets=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((assets/'metrics.json').read_text(encoding='utf-8'))
        boundaries={'regularized_objective_per_datum':('<=',config['regularized_objective_per_datum_boundary']),'observation_relative_error':('<=',config['observation_relative_error_boundary'])}
        if not all(np.isfinite(v) for _,v in boundaries.values()):raise ValueError('nonfinite metric boundary')
        report['thresholds']={k:{'comparison':op,'boundary':v} for k,(op,v) in boundaries.items()}
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as d:y,m=d['measurements'],d['spectra'][0]
        with np.load(prediction,allow_pickle=False) as d:
            if 'reconstruction' not in d:raise ValueError('missing required key reconstruction')
            x=d['reconstruction'];validate(x)
        result={'regularized_objective_per_datum':regularized_score(x,y,m),'observation_relative_error':metrics(x,x,y,m)['observation_relative_error'] if np.ptp(x)>0 else 1.}
        valid_physical=bool(np.all(x>=0) and np.all(x<=1+1e-6) and np.max(x)>=1-1e-6 and np.all(x[:,[0,-1]]==0) and np.all(x[:,:,[0,-1]]==0))
        passed=valid_physical and all(result[k]>=v if op=='>=' else result[k]<=v for k,(op,v) in boundaries.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=result)
        if not valid_physical:report['errors'].append('require nonnegative max-one abundance and zero exterior border')
        return report,0 if passed else 1
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
        return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args(argv);report,code=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:
        print(f'report write failed: {exc}',file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
