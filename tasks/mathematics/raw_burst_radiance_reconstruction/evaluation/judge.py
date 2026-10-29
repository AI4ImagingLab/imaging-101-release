"""Independent strict NPZ submission judge."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
def relative_l2(x,y):
    scale=float(np.max(np.abs(y)))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid zero/nonfinite reference norm')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        value=float(np.linalg.norm((x.astype(np.complex128)-y)/scale)/np.linalg.norm(y/scale))
    if not np.isfinite(value):raise ValueError('Metric exceeds finite numerical range')
    return value
def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'))
        contracts=config['outputs']
        if not isinstance(contracts,dict) or not contracts:raise ValueError('Empty output definitions')
        with np.load(ROOT/'baseline_reference.npz',allow_pickle=False) as f:refs={k:f[k].copy() for k in contracts}
        with np.load(prediction,allow_pickle=False) as f:preds={k:f[k].copy() for k in contracts}
        for key,spec in contracts.items():
            boundary=spec['relative_l2_boundary']
            if type(boundary) not in (int,float) or not np.isfinite(boundary) or boundary<=0:raise ValueError('Invalid boundary for '+key)
            report['thresholds'][key+'_relative_l2']={'comparison':'<=','boundary':boundary}
            for array,role in [(refs[key],'reference'),(preds[key],'prediction')]:
                if list(array.shape)!=spec['shape']:raise ValueError(role+' '+key+' shape must be '+str(spec['shape']))
                if array.dtype!=np.dtype(spec['dtype']):raise ValueError(role+' '+key+' dtype must be '+spec['dtype'])
                if not np.isfinite(array).all():raise ValueError(role+' '+key+' contains NaN or Infinity')
        values={k+'_relative_l2':relative_l2(preds[k],refs[k]) for k in contracts}
        passed=all(v<=report['thresholds'][k]['boundary'] for k,v in values.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values)
        return report,0 if passed else 1
    except Exception as exc:report['metrics']={};report['errors']=[str(exc)];return report,2
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
