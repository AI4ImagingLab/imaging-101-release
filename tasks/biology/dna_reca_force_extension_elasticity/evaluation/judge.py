"""Independent physical-unit comparison, strict submission validation."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'persistence_length':(1,1),'stretch_modulus':(1,1),'contour_lengths':(1,2),'predicted_force_control':(1,884),'predicted_force_reca':(1,1030)}
def load_arrays(path):
    if not Path(path).is_file():raise ValueError('file not found: '+str(path))
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!=set(SHAPES):raise ValueError('required exact output keys missing or extra')
        out={k:f[k].copy() for k in SHAPES}
    for k,a in out.items():
        if a.shape!=SHAPES[k]:raise ValueError('invalid shape '+k)
        if a.dtype!=np.dtype('float64'):raise ValueError('required float64 '+k)
        if not np.isfinite(a).all():raise ValueError('nonfinite '+k)
    return out
def score(pred,ref,observed,bounds):
    metrics={k:float(np.max(abs(pred[k]-ref[k]))) for k in SHAPES}
    for n in ['control','reca']:
        f=observed['force_'+n][0];mask=f<=30
        metrics['observation_rmse_'+n]=float(np.sqrt(np.mean((pred['predicted_force_'+n][0]-f[mask])**2)))
    if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Undefined or overflowing evaluation metric')
    return metrics,all(metrics[k]<=bounds[k] for k in metrics)
def evaluate(path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        bounds=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))['boundaries']
        if set(bounds)!=set(SHAPES)|{'observation_rmse_control','observation_rmse_reca'} or any(not np.isfinite(v) or v<=0 for v in bounds.values()):raise ValueError('invalid metric configuration')
        report['thresholds']={k:{'comparison':'<=','boundary':v} for k,v in bounds.items()}
        reference=HERE/'ground_truth.npz'
        if not reference.exists():reference=HERE/'baseline_reference.npz'
        ref=load_arrays(reference);pred=load_arrays(path)
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as f:observed={k:f[k] for k in ['force_control','force_reca']}
        values,passed=score(pred,ref,observed,bounds);report.update(status='pass' if passed else 'fail',passed=passed,metrics=values)
        return report,0 if passed else 1
    except Exception as exc:report.update(status='error',passed=False,metrics={},errors=[str(exc)]);return report,2
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();r,code=evaluate(a.prediction)
    try:
        dest=Path(a.report);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
