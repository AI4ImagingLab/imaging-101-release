"""Strict standalone evaluator; never imports a solver."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'enthalpy':(1,1),'association_constant':(1,1),'effective_site_factor':(1,1),
        'dilution_slope':(1,6),'dilution_offset':(1,6),'predicted_heats':(1,6,14)}
def load_arrays(path):
    if not Path(path).is_file(): raise ValueError('file not found: '+str(path))
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!=set(SHAPES): raise ValueError('required exact keys: '+','.join(SHAPES))
        out={k:f[k].copy() for k in SHAPES}
    for k,a in out.items():
        if a.shape!=SHAPES[k]:raise ValueError('shape mismatch: '+k)
        if a.dtype!=np.dtype('float64'):raise ValueError('dtype must be float64: '+k)
        if not np.isfinite(a).all():raise ValueError('nonfinite values: '+k)
    return out
def score(pred,ref,heats,bounds):
    values={k:float(np.max(np.abs(pred[k]-ref[k]))) for k in SHAPES}
    values['observation_rmse']=float(np.sqrt(np.mean((pred['predicted_heats'][0,:,1:]-heats[0,:,1:])**2)))
    if not all(np.isfinite(v) for v in values.values()):raise ValueError('Undefined or overflowing evaluation metric')
    return values,all(values[k]<=bounds[k] for k in values)
def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        bounds=config['boundaries']
        if set(bounds)!=set(SHAPES)|{'observation_rmse'} or any(not np.isfinite(v) or v<=0 for v in bounds.values()):raise ValueError('invalid evaluator boundaries')
        report['thresholds']={k:{'comparison':'<=','boundary':v} for k,v in bounds.items()}
        reference=HERE/'ground_truth.npz'
        if not reference.exists():reference=HERE/'baseline_reference.npz'
        ref=load_arrays(reference)
        pred=load_arrays(prediction)
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as f:heats=f['heats']
        values,passed=score(pred,ref,heats,bounds)
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=values)
        return report,0 if passed else 1
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},errors=[str(exc)]);return report,2
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print('report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
