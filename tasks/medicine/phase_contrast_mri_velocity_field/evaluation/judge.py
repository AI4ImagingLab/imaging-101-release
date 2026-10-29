"""Independent NPZ evaluator: all3velocity directions, physical phase and mask."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ASSETS=Path(__file__).resolve().parent
SHAPE=(1,27,3,54,67,100)
METRIC_NAMES=('ncc','nrmse','x_ncc','x_nrmse','y_ncc','y_nrmse','z_ncc','z_nrmse','phase_phasor_rms','outside_mask_max')

def read_field(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('Missing reconstruction')
        a=z['reconstruction']
    if a.shape!=SHAPE or a.dtype!=np.float32 or not np.isfinite(a).all():
        raise ValueError('Expected finite real float32 shape '+str(SHAPE)+' in '+str(path))
    return a[0]

def compare(a,b):
    x=np.asarray(a,dtype=np.float64).ravel();r=np.asarray(b,dtype=np.float64).ravel()
    spread=float(np.ptp(r));rn=np.linalg.norm(r);xn=np.linalg.norm(x)
    if spread<=0 or rn<=0:raise ValueError('Undefined constant/zero evaluator reference')
    cosine=float(x@r/(xn*rn)) if xn>0 else 0.
    error=float(np.sqrt(np.mean((x-r)**2))/spread)
    return cosine,error

def observation_metrics(field):
    """Independent real/imaginary phase and explicit phasor forward prediction.

    Use complex128 measurement arithmetic independent of source complex64
    extraction. Static background is fixed by public arithmetic phase mean;
    it is not estimated from a submission or hidden velocity reference.
    """
    with np.load(ASSETS.parent/'data/raw_data.npz',allow_pickle=False) as z:
        obs=z['encoding_images']
    if obs.shape!=(1,27,4,54,67,100) or obs.dtype!=np.complex64 or not np.isfinite(obs).all():raise ValueError('Malformed public acquisition')
    obs=obs[0];ref=obs[:,0].astype(np.complex128);magnitude=np.abs(obs[:,0])
    mask=magnitude>.1*magnitude.max();total=0.;n=0;outside=0.
    for d in range(3):
        product=obs[:,d+1].astype(np.complex128)*np.conj(ref)
        angle=np.arctan2(product.imag,product.real);background=angle.mean(axis=0)
        prediction=np.exp(1j*(np.pi*field[:,d].astype(np.float64)/.16+background[None]))
        measured=np.exp(1j*angle);error=np.abs(prediction[mask]-measured[mask])
        total+=float(error@error);n+=error.size
        if (~mask).any():outside=max(outside,float(np.max(np.abs(field[:,d][~mask]))))
    if n==0:raise ValueError('Empty public magnitude mask')
    return dict(phase_phasor_rms=float(np.sqrt(total/n)),outside_mask_max=outside)

def passes(metrics,thresholds):
    return all(metrics[k]>=v['boundary'] if v['comparison']=='>=' else metrics[k]<=v['boundary'] for k,v in thresholds.items())

def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((ASSETS/'metrics.json').read_text(encoding='utf8'));thresholds={}
        for name in METRIC_NAMES:
            b=config[name+'_boundary'];is_cosine=name.endswith('ncc')
            if isinstance(b,bool) or not isinstance(b,(int,float)) or not np.isfinite(b) or (is_cosine and not -1<=b<=1) or (not is_cosine and b<0):raise ValueError('Invalid boundary '+name)
            thresholds[name]=dict(comparison='>=' if is_cosine else '<=',boundary=b)
        a=read_field(Path(prediction));truth=ASSETS/'ground_truth.npz'
        b=read_field(truth if truth.exists() else ASSETS/'baseline_reference.npz')
        with np.errstate(over='raise',divide='raise',invalid='raise'):
            n,r=compare(a,b);metrics=dict(ncc=n,nrmse=r)
            for k,label in enumerate('xyz'):
                n,r=compare(a[:,k],b[:,k]);metrics[label+'_ncc']=n;metrics[label+'_nrmse']=r
            metrics.update(observation_metrics(a))
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite calculated metric')
        ok=passes(metrics,thresholds)
        report.update(status='pass' if ok else 'fail',passed=bool(ok),metrics=metrics,thresholds=thresholds)
        return report,0 if ok else 1
    except Exception as exc:report['errors']=[f'{type(exc).__name__}: {exc}'];return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);a=parser.parse_args(argv)
    report,code=evaluate(a.prediction)
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
