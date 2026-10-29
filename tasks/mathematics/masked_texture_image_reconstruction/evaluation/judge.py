"""Standalone evaluator, with no imports of reference or submitted solvers."""
from pathlib import Path
import argparse,json,sys,math
import numpy as np
ASSETS=Path(__file__).resolve().parent
SHAPE=(1,240,320)
def score(a,b,mask):
    spread=float(np.ptp(b));normb=float(np.linalg.norm(b))
    if spread<=0 or normb<=0:raise ValueError('Degenerate hidden reference')
    norma=float(np.linalg.norm(a));ncc=float(np.vdot(a,b)/(norma*normb)) if norma else 0.
    return dict(ncc=ncc,nrmse=float(np.sqrt(np.mean((a-b)**2))/spread),missing_nrmse=float(np.sqrt(np.mean((a[~mask]-b[~mask])**2))/spread))
def load(path,key,shape,dtype):
    with np.load(path,allow_pickle=False) as z:
        if key not in z:raise ValueError(f'Missing key {key}')
        a=z[key]
    if a.shape!=shape or a.dtype!=dtype:raise ValueError(f'Invalid {key}: expected {shape}, {dtype}')
    if not np.isfinite(a).all():raise ValueError(f'Nonfinite {key}')
    return a
def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        cfg=json.loads((ASSETS/'metrics.json').read_text(encoding='utf8'))
        thresholds={}
        for name in ['ncc','nrmse','missing_nrmse','known_pixel_max_error']:
            boundary=cfg[name+'_boundary']
            if isinstance(boundary,bool) or not isinstance(boundary,(int,float)) or not math.isfinite(boundary):raise ValueError('Invalid numeric boundary '+name)
            if (name=='ncc' and not -1<=boundary<=1) or (name!='ncc' and boundary<0):raise ValueError('Invalid boundary range '+name)
            thresholds[name]=dict(comparison='>=' if name=='ncc' else '<=',boundary=boundary)
        reference=ASSETS/'ground_truth.npz'
        if not reference.exists():reference=ASSETS/'baseline_reference.npz'
        truth=load(reference,'reconstruction',SHAPE,np.dtype('float64'))
        raw=ASSETS.parent/'data/raw_data.npz'
        y=load(raw,'observations',SHAPE,np.dtype('float64'));mask=load(raw,'mask',SHAPE,np.dtype('bool'))
        if not mask.any() or mask.all():raise ValueError('Degenerate observation mask')
        a=load(Path(prediction),'reconstruction',SHAPE,np.dtype('float64'))
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            measured=score(a,truth,mask)
            measured['known_pixel_max_error']=float(np.max(np.abs(a[mask]-y[mask])))
        if not all(math.isfinite(v) for v in measured.values()):raise ValueError('Undefined computed metrics')
        passed=all(measured[k]>=v['boundary'] if v['comparison']=='>=' else measured[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[f'{type(exc).__name__}: {exc}']
        return report,2
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:
        print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())

