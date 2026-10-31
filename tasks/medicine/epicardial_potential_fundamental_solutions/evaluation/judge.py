"""Standalone numerical-reference ECGI evaluator; no solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,247)}
def load_arrays(path):
    if not Path(path).is_file():raise ValueError('file not found: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(SHAPES):raise ValueError('required exact output key reconstruction')
        a=z['reconstruction'].copy()
    if a.shape!=(1,247):raise ValueError('expected shape (1,247)')
    if a.dtype!=np.dtype('float64'):raise ValueError('expected real float64')
    if not np.isfinite(a).all():raise ValueError('nonfinite prediction')
    return a
def score(pred,ref):
    r=ref.ravel();p=pred.ravel();scale=np.ptp(r);rn=np.linalg.norm(r);pn=np.linalg.norm(p)
    if scale<=0 or rn<=0:raise ValueError('degenerate evaluator reference')
    values={'ncc':float(np.dot(p,r)/(pn*rn)) if pn>0 else 0.,
       'nrmse':float(np.sqrt(np.mean((p-r)**2))/scale),
       'max_abs_error':float(np.max(np.abs(p-r)))}
    if not all(np.isfinite(v) for v in values.values()):raise ValueError('Undefined or overflowing evaluation metric')
    return values
def evaluate(path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        c=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        bounds={k:c[k+'_boundary'] for k in ['ncc','nrmse','max_abs_error']}
        if any(not np.isfinite(x) or x<=0 for x in bounds.values()) or bounds['ncc']>1:raise ValueError('invalid metric boundaries')
        report['thresholds']={k:{'comparison':'>=' if k=='ncc' else '<=','boundary':v} for k,v in bounds.items()}
        reference=HERE/'ground_truth.npz'
        if not reference.exists():reference=HERE/'baseline_reference.npz'
        ref=load_arrays(reference);pred=load_arrays(path);m=score(pred,ref)
        passed=m['ncc']>=bounds['ncc'] and m['nrmse']<=bounds['nrmse'] and m['max_abs_error']<=bounds['max_abs_error']
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=m);return report,0 if passed else 1
    except Exception as exc:report.update(status='error',passed=False,metrics={},errors=[str(exc)]);return report,2
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args();report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
