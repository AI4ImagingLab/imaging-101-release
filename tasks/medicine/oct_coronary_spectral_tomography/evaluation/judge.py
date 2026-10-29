"""Standalone evaluator: never imports any solver code."""
from pathlib import Path
import json,argparse,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
SHAPE=(1,2,2048,1024)
def relative_l2(x,y):
    scale=float(np.max(np.abs(y)))
    if not np.isfinite(scale) or scale<=0: raise ValueError('Reference norm is undefined')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        value=float(np.linalg.norm((x.astype(np.complex128)-y)/scale)/np.linalg.norm(y/scale))
    if not np.isfinite(value): raise ValueError('Nonfinite metric')
    return value
def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'))
        boundary=config['relative_l2_boundary']
        if type(boundary) not in (int,float) or not np.isfinite(boundary) or boundary<=0: raise ValueError('Invalid boundary')
        report['thresholds']={'relative_l2':{'comparison':'<=','boundary':boundary}}
        with np.load(ROOT/'baseline_reference.npz',allow_pickle=False) as f: ref=f['reconstruction'].copy()
        if ref.shape!=SHAPE or ref.dtype!=np.complex64 or not np.isfinite(ref).all(): raise ValueError('Invalid hidden reference')
        with np.load(prediction,allow_pickle=False) as f: pred=f['reconstruction'].copy()
        if pred.shape!=SHAPE: raise ValueError('Expected shape '+str(SHAPE))
        if pred.dtype!=np.complex64: raise ValueError('Expected complex64 dtype')
        if not np.isfinite(pred).all(): raise ValueError('Prediction contains NaN or Infinity')
        value=relative_l2(pred,ref)
        report['metrics']={'relative_l2':value}; report['passed']=bool(value<=boundary)
        report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as exc:
        report['metrics']={};report['errors']=[str(exc)];return report,2
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc: print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
