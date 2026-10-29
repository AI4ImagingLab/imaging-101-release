"""Standalone evaluator: never imports submitted code or reference solver."""
from pathlib import Path
import argparse,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent
KEYS=('reconstruction','relative_flow','relative_volume','transit_time')

def score_arrays(estimate,reference):
    """Cosine NCC without centering; RMS error divided by reference range."""
    x=np.asarray(estimate).reshape(-1);r=np.asarray(reference).reshape(-1)
    if x.shape!=r.shape or not np.isfinite(x).all() or not np.isfinite(r).all():raise ValueError('Invalid metric arrays')
    lo=float(r.min());hi=float(r.max())
    if hi<=lo:raise ValueError('Undefined range normalization for constant reference')
    dot=xx=rr=error=0.
    for start in range(0,len(x),262144):
        a=x[start:start+262144];b=r[start:start+262144]
        dot+=float(a@b);xx+=float(a@a);rr+=float(b@b);error+=float(np.sum((a-b)**2))
    if rr<=0:raise ValueError('Undefined zero-norm reference')
    product=xx*rr
    if xx>0 and np.isfinite(product) and product>0:ncc=dot/np.sqrt(product)
    elif xx>0 and np.isfinite(xx) and np.isfinite(rr):ncc=(dot/np.sqrt(xx))/np.sqrt(rr)
    else:ncc=0. if xx==0 else float('nan')
    nrmse=np.sqrt(error/len(x))/(hi-lo)
    if not np.isfinite(ncc) or not np.isfinite(nrmse):raise ValueError('Metric overflow')
    return ncc,nrmse

def _forward_error(residue,raw_path):
    with np.load(raw_path,allow_pickle=False) as z:
        signal=z['signal'][0];support=z['support'][0];a=z['arterial_input'][0]
    n=len(a);col=np.zeros(2*n);col[0]=a[0]
    col[1:n-1]=(a[:-2]+4*a[1:-1]+a[2:])/6;col[n-1]=(a[-2]+4*a[-1])/6;col[n]=a[-1]/6
    matrix=col[(np.arange(n)[:,None]-np.arange(2*n)[None,:])%(2*n)]*1.55
    error=norm=0.
    for s in range(len(signal)):
        m=support[s]
        if not m.any():continue
        observed=signal[s][m].astype(np.float64)
        if np.any(observed<=0):raise ValueError('Evaluator observation support has invalid logarithm input')
        average=observed.mean(axis=0);count=3
        while True:
            b=average[:count].mean()
            if abs((b-average[count])/b)<.05:count+=1
            else:count-=1;break
            if count==12:count-=1;break
        y=-np.log(observed/observed[:,:count].mean(axis=1,keepdims=True))/.025
        estimate=residue[0,s][m]@matrix.T
        error+=float(np.sum((estimate-y)**2));norm+=float(np.sum(y*y))
    if norm<=0:raise ValueError('Undefined observation norm')
    return float(np.sqrt(error/norm))

def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        for key in ('ncc_boundary','nrmse_boundary','forward_relative_l2_boundary'):
            v=config[key]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v):raise ValueError('Invalid finite numeric boundary: '+key)
        if not -1<=config['ncc_boundary']<=1 or config['nrmse_boundary']<=0 or config['forward_relative_l2_boundary']<=0:raise ValueError('Invalid metric boundaries')
        specs=config['required_outputs']
        if set(specs)!=set(KEYS):raise ValueError('Evaluator required-output definition mismatch')
        reference=HERE/'ground_truth.npz'
        if not reference.exists():reference=HERE/'baseline_reference.npz'
        if not reference.is_file():raise ValueError('Unavailable hidden reference asset')
        if not Path(prediction).is_file():raise ValueError('Prediction file does not exist')
        with np.load(prediction,allow_pickle=False) as submitted, np.load(reference,allow_pickle=False) as expected:
            for key in KEYS:
                if key not in submitted:raise ValueError('Missing required prediction key: '+key)
                x=submitted[key];spec=specs[key]
                if list(x.shape)!=spec['shape']:raise ValueError('Wrong prediction shape: '+key)
                if x.dtype!=np.dtype(spec['dtype']) or np.iscomplexobj(x):raise ValueError('Wrong prediction dtype: '+key)
                if not np.isfinite(x).all():raise ValueError('Nonfinite prediction: '+key)
                if key not in expected:raise ValueError('Missing hidden reference key: '+key)
                r=expected[key]
                if r.shape!=x.shape or r.dtype!=x.dtype or not np.isfinite(r).all():raise ValueError('Invalid hidden reference contract: '+key)
            values={};boundaries={}
            with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as public:support=public['support']
            for key in KEYS:
                x=submitted[key]
                if np.any(x[~support]!=0):raise ValueError('Required zero outside acquisition support: '+key)
                ncc,nrmse=score_arrays(x,expected[key]);values[key+'_ncc']=float(ncc);values[key+'_nrmse']=float(nrmse)
                boundaries[key+'_ncc']=dict(comparison='>=',boundary=config['ncc_boundary'])
                boundaries[key+'_nrmse']=dict(comparison='<=',boundary=config['nrmse_boundary'])
            values['forward_relative_l2']=_forward_error(submitted['reconstruction'],HERE.parent/'data/raw_data.npz')
            boundaries['forward_relative_l2']=dict(comparison='<=',boundary=config['forward_relative_l2_boundary'])
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Nonfinite measured metric')
        passed=all(values[k]>=v['boundary'] if v['comparison']=='>=' else values[k]<=v['boundary'] for k,v in boundaries.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values,thresholds=boundaries)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[f'{type(exc).__name__}: {exc}'];return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report-write failure: '+str(exc),file=sys.stderr);return 2
    print(json.dumps(report,allow_nan=False));return code

if __name__=='__main__':sys.exit(main())
