"""Standalone strict NPZ evaluator; imports no solver code."""
import argparse,json,sys
from pathlib import Path
import numpy as np
SHAPES={'rates':(1,11),'reconstruction':(1,150,15),'charge':(1,150)}

def load(path):
    if not path.is_file():raise ValueError(f'Missing file: {path}')
    with np.load(path,allow_pickle=False) as f:
        out={}
        for key,shape in SHAPES.items():
            if key not in f:raise ValueError(f'Missing required key: {key}')
            a=f[key]
            if a.shape!=shape:raise ValueError(f'{key} shape must be {shape}')
            if a.dtype!=np.float64:raise ValueError(f'{key} dtype must be float64')
            if not np.all(np.isfinite(a)):raise ValueError(f'{key} contains NaN/Inf')
            out[key]=a
    return out

def score(pred,ref,config):
    metrics={};thresholds={}
    for i in range(11):
        name=f'rate_{i}_absolute_error'
        metrics[name]=float(abs(pred['rates'][0,i]-ref['rates'][0,i]))
        boundary=config['rates_absolute_boundary']+config['rates_relative_boundary']*abs(float(ref['rates'][0,i]))
        thresholds[name]={'comparison':'<=','boundary':boundary}
    for i in range(15):
        a=pred['reconstruction'][0,:,i];b=ref['reconstruction'][0,:,i]
        scale=float(np.ptp(b))
        if scale<=0:raise ValueError('Degenerate reference species')
        name=f'species_{i}_nrmse';metrics[name]=float(np.sqrt(np.mean((a-b)**2))/scale)
        thresholds[name]={'comparison':'<=','boundary':config['concentration_nrmse_boundary']}
    a=pred['charge'];b=ref['charge'];scale=float(np.ptp(b))
    if scale<=0:raise ValueError('Degenerate reference charge')
    metrics['charge_nrmse']=float(np.sqrt(np.mean((a-b)**2))/scale)
    thresholds['charge_nrmse']={'comparison':'<=','boundary':config['charge_nrmse_boundary']}
    for name,d in thresholds.items():
        if not np.isfinite(d['boundary']) or d['boundary']<=0:raise ValueError(f'Invalid boundary {name}')
    if any(not np.isfinite(value) for value in metrics.values()):
        raise ValueError('Nonfinite metric result (numerical overflow)')
    return metrics,thresholds

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args(argv)
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]};code=2
    try:
        base=Path(__file__).resolve().parent
        config=json.loads((base/'metrics.json').read_text(encoding='utf8'))
        truth=base/'ground_truth.npz';ref=load(truth if truth.exists() else base/'baseline_reference.npz')
        pred=load(Path(args.prediction))
        metrics,thresholds=score(pred,ref,config)
        passed=all(v<=thresholds[k]['boundary'] for k,v in metrics.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=thresholds);code=0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)]
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print(f'Report-write failure: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
