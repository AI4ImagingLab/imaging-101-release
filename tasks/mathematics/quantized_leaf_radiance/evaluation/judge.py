"""Standalone independent truth and public-interval scoring."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPE=(1,512,512,3)

def read_image(path):
    with np.load(path,allow_pickle=False) as f:
        if 'reconstruction' not in f:raise ValueError('missing required reconstruction key')
        a=f['reconstruction']
    if a.shape!=SHAPE:raise ValueError('expected batch,row,column,RGB shape '+str(SHAPE))
    if a.dtype!=np.float32:raise ValueError('expected float32 reconstruction')
    if not np.isfinite(a).all():raise ValueError('non-finite image')
    if np.any((a<0)|(a>1)):raise ValueError('image must have normalized intensity in [0,1]')
    return a.astype(np.float64)

def compare(estimate,reference):
    x=estimate.ravel();r=reference.ravel();xn=np.linalg.norm(x);rn=np.linalg.norm(r);span=np.ptp(r)
    if rn==0 or span==0:raise ValueError('degenerate hidden reference')
    return {'ncc':float(x@r/(xn*rn)) if xn else 0.,'nrmse':float(np.sqrt(np.mean((x-r)**2))/span)}

def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        c=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        for name in ['ncc','nrmse','interval_violation']:
            b=c[name+'_boundary']
            if isinstance(b,bool) or not isinstance(b,(int,float)) or not np.isfinite(b):raise ValueError('invalid metric boundary '+name)
            if name=='ncc' and not 0<b<=1:raise ValueError('invalid NCC boundary')
            if name!='ncc' and b<=0:raise ValueError('nonpositive error boundary')
            report['thresholds'][name]={'comparison':'>=' if name=='ncc' else '<=','boundary':b}
        actual=HERE/'ground_truth.npz';reference=read_image(actual if actual.exists() else HERE/'baseline_reference.npz')
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as f:
            lo=f['lower'];hi=f['upper']
            if lo.shape!=SHAPE or hi.shape!=SHAPE or lo.dtype!=np.float32 or hi.dtype!=np.float32 or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo>hi):raise ValueError('evaluator public interval asset invalid')
        x=read_image(prediction);m=compare(x,reference)
        m['interval_violation']=max(0.,float(np.max(lo-x)),float(np.max(x-hi)))
        if not np.isfinite(list(m.values())).all():raise ValueError('non-finite metric arithmetic')
        passed=m['ncc']>=c['ncc_boundary'] and m['nrmse']<=c['nrmse_boundary'] and m['interval_violation']<=c['interval_violation_boundary']
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=m)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)];return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
