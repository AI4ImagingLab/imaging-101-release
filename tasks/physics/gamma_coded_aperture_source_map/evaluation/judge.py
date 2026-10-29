"""Standalone deterministic evaluator, independent of the solver."""
from pathlib import Path
import numpy as np
import json,argparse,sys
ASSETS=Path(__file__).resolve().parent;SHAPE=(1,126,126)
def load_prediction(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('Missing reconstruction')
        a=z['reconstruction']
    if a.shape!=SHAPE or a.dtype!=np.float32:raise ValueError('Require float32 reconstruction(1,126,126)')
    if not np.isfinite(a).all():raise ValueError('Nonfinite reconstruction')
    return a
def score(x,r):
    a=x.astype(np.float64).ravel();b=r.astype(np.float64).ravel();span=float(np.ptp(b));norm=float(np.linalg.norm(b))
    if not span>0 or not norm>0:raise ValueError('Degenerate hidden reference')
    na=float(np.linalg.norm(a));return {'ncc':float(np.dot(a,b)/(na*norm)) if na>0 else 0.,'nrmse':float(np.sqrt(np.mean((a-b)**2))/span)}
def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((ASSETS/'metrics.json').read_text(encoding='utf8'));nb=cfg['ncc_boundary'];eb=cfg['nrmse_boundary']
        if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) for v in [nb,eb]) or not 0<nb<=1 or not 0<eb<1:raise ValueError('Invalid metric boundaries')
        p=ASSETS/'ground_truth.npz';r=load_prediction(p if p.exists() else ASSETS/'baseline_reference.npz');x=load_prediction(prediction)
        with np.errstate(over='raise',invalid='raise',divide='raise'):m=score(x,r)
        if not all(np.isfinite(v) for v in m.values()):raise ValueError('Nonfinite metric')
        ok=m['ncc']>=nb and m['nrmse']<=eb;report.update(status='pass' if ok else 'fail',passed=bool(ok),metrics=m,thresholds={'ncc':{'comparison':'>=','boundary':nb},'nrmse':{'comparison':'<=','boundary':eb}});return report,0 if ok else 1
    except Exception as e:report['errors']=[f'{type(e).__name__}: {e}'];return report,2
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args(argv);report,code=evaluate(args.prediction)
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as e:print(f'Cannot write report: {e}',file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
