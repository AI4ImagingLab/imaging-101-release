"""Independent all-quantity, physical-unit DCS evaluator."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'beta':(1,390),'flow_index':(1,390),'predicted_g2':(1,390,199)}
NAMES=['beta_max_abs_error','flow_index_max_relative_error','predicted_g2_max_abs_error','masked_g2_rmse','worst_curve_g2_rmse']
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
def observation_support(raw):
    c=raw['countrate'][0];g=raw['g2'][0];tau=raw['lag_times'][0]
    weighted=(g*c[:,:,None]).sum(axis=1)/c.sum(axis=1)[:,None]
    masks=np.zeros(weighted.shape,bool)
    start=np.searchsorted(tau,1e-7,side='right');end=np.searchsorted(tau,1e-3,side='right')
    for i,y in enumerate(weighted):
        above=np.flatnonzero(y>1.13);stop=min(end,int(above[-1])+1) if above.size else end
        masks[i,start:stop]=True
    return weighted,masks
def score(pred,ref,weighted,mask):
    if np.any(ref['flow_index']<=0):raise ValueError('reference flow index must be positive')
    diff=pred['predicted_g2'][0]-weighted
    values={'beta_max_abs_error':float(np.max(np.abs(pred['beta']-ref['beta']))),
       'flow_index_max_relative_error':float(np.max(np.abs(pred['flow_index']/ref['flow_index']-1))),
       'predicted_g2_max_abs_error':float(np.max(np.abs(pred['predicted_g2']-ref['predicted_g2']))),
       'masked_g2_rmse':float(np.sqrt(np.mean(diff[mask]**2))),
       'worst_curve_g2_rmse':float(max(np.sqrt(np.mean(e[m]**2)) for e,m in zip(diff,mask)))}
    if not all(np.isfinite(v) for v in values.values()):raise ValueError('Undefined or overflowing evaluation metric')
    return values
def evaluate(path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        bounds={k:config[k+'_boundary'] for k in NAMES}
        if any(not np.isfinite(v) or v<=0 for v in bounds.values()):raise ValueError('invalid metric configuration')
        report['thresholds']={k:{'comparison':'<=','boundary':v} for k,v in bounds.items()}
        reference=HERE/'ground_truth.npz'
        if not reference.exists():reference=HERE/'baseline_reference.npz'
        ref=load_arrays(reference);pred=load_arrays(path)
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as raw:weighted,mask=observation_support(raw)
        values=score(pred,ref,weighted,mask);passed=all(values[k]<=bounds[k] for k in NAMES)
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=values)
        return report,0 if passed else 1
    except Exception as exc:report.update(status='error',passed=False,metrics={},errors=[str(exc)]);return report,2
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();r,code=evaluate(a.prediction)
    try:
        dest=Path(a.report);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
