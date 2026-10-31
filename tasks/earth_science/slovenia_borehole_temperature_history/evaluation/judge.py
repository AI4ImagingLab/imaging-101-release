"""Standalone NPZ evaluator. Never imports reconstruction code."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import erfc

HERE=Path(__file__).resolve().parent
SHAPE=(1,7,21)

def load_prediction(path):
    if not Path(path).is_file(): raise ValueError('prediction file does not exist')
    with np.load(path,allow_pickle=False) as f:
        if 'reconstruction' not in f.files: raise ValueError('missing reconstruction key')
        x=f['reconstruction']
    if x.shape!=SHAPE: raise ValueError(f'expected shape {SHAPE}, got {x.shape}')
    if x.dtype not in (np.dtype('float32'),np.dtype('float64')): raise ValueError('expected real float32 or float64')
    if not np.isfinite(x).all(): raise ValueError('prediction contains NaN or Infinity')
    return x.astype(np.float64,copy=False)

def agreement(x,reference):
    span=float(np.ptp(reference))
    norm=float(np.linalg.norm(reference))
    if span<=0 or norm<=0: raise ValueError('degenerate reference')
    xn=float(np.linalg.norm(x))
    return {'ncc':float(np.sum(x*reference)/(xn*norm)) if xn else 0.,'nrmse':float(np.sqrt(np.mean((x-reference)**2))/span)}

def evaluate(prediction,assets=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        x=load_prediction(prediction)
        assets=Path(assets)
        config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        reference_path=assets/'ground_truth.npz'
        if not reference_path.exists(): reference_path=assets/'baseline_reference.npz'
        reference=load_prediction(reference_path)
        metrics=agreement(x,reference)
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as d:
            depth=d['depth'][0];temperature=d['temperature'][0]
            counts=d['counts'][0];offset=0;residuals=[]
            times=30.*np.arange(1,22)
            for i,count in enumerate(counts):
                z=depth[offset:offset+count];y=temperature[offset:offset+count];offset+=count
                selected=(z>=200)&(z<=300)
                design=np.column_stack([np.ones(selected.sum()),z[selected]])
                intercept,slope=np.linalg.lstsq(design,y[selected],rcond=None)[0]
                response=erfc(z[:,None]/(2*np.sqrt(31.5576*times[None,:])))
                response[:,1:]-=response[:,:-1].copy()
                residuals.append(np.sqrt(np.mean((response@x[0,i]+intercept+slope*z-y)**2)))
            metrics['max_temperature_rmse']=float(max(residuals))
        comparisons={'ncc':'>=','nrmse':'<=','max_temperature_rmse':'<='}
        thresholds={k:{'comparison':op,'boundary':config[k+'_boundary']} for k,op in comparisons.items()}
        if not all(np.isfinite(v) for v in metrics.values()): raise ValueError('nonfinite computed metric')
        if not all(np.isfinite(t['boundary']) for t in thresholds.values()): raise ValueError('nonfinite threshold')
        passed=all(metrics[k]>=t['boundary'] if t['comparison']=='>=' else metrics[k]<=t['boundary'] for k,t in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds)
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},errors=[str(exc)])
    return report

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--prediction',required=True)
    p.add_argument('--report',required=True)
    args=p.parse_args(argv)
    report=evaluate(args.prediction)
    try:
        dest=Path(args.report);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:
        print(f'report write failed: {exc}',file=sys.stderr)
        return 2
    return {'pass':0,'fail':1,'error':2}[report['status']]

if __name__=='__main__': sys.exit(main())
