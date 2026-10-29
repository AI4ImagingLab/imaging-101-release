"""Independent strict component and measured additive-consistency evaluator."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ASSETS=Path(__file__).resolve().parent


def relative_error(x,r):
    norm=float(np.linalg.norm(r.ravel()))
    if not np.isfinite(norm) or norm<=0:raise ValueError('Invalid reference norm')
    v=float(np.linalg.norm((x-r).ravel())/norm)
    if not np.isfinite(v):raise ValueError('Nonfinite metric')
    return v


def read_arrays(path,specs):
    with np.load(path,allow_pickle=False) as z:
        a={k:z[k] for k in specs}
    for k,s in specs.items():
        if a[k].shape!=tuple(s['shape']) or a[k].dtype!=np.dtype(s['dtype']):raise ValueError('Invalid shape/dtype: '+k)
        if not np.isfinite(a[k]).all():raise ValueError('Nonfinite array: '+k)
    return a


def evaluate(prediction,asset_dir=ASSETS):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        assets=Path(asset_dir);c=json.loads((assets/'metrics.json').read_text(encoding='utf8'));specs=c['outputs']
        if set(specs)!={'reconstruction','sparse_component'}:raise ValueError('Invalid output configuration')
        bounds={k+'_relative_l2':s['relative_l2_boundary'] for k,s in specs.items()}
        bounds['additive_consistency_relative']=c['additive_consistency_relative_boundary']
        if any(not np.isfinite(v) or v<=0 for v in bounds.values()):raise ValueError('Invalid metric boundary')
        path=assets/'ground_truth.npz'
        if not path.exists():path=assets/'baseline_reference.npz'
        ref=read_arrays(path,specs);pred=read_arrays(prediction,specs)
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as z:frames=z['frames']
        if frames.shape!=(1,84,240,320) or frames.dtype!=np.uint8:raise ValueError('Invalid evaluator observations')
        values={k+'_relative_l2':relative_error(pred[k],ref[k]) for k in specs}
        d=frames.astype(np.float64)/255
        values['additive_consistency_relative']=relative_error(pred['reconstruction']+pred['sparse_component'],d)
        passed=all(values[k]<=bounds[k] for k in bounds)
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=values,thresholds={k:{'comparison':'<=','boundary':v} for k,v in bounds.items()})
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)];return report,2


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    r,code=evaluate(a.prediction)
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code


if __name__=='__main__':sys.exit(main())
