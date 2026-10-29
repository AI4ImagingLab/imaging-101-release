"""Standalone NPZ evaluator. Never imports reconstruction code."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
SHAPE=(1,190,120,41)

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
            geometry=d['geometry'][0]
            channels=d['good_channels'][0].astype(int)-1
            scale=1/geometry.max()
            matrix=(geometry*scale)[channels].reshape(len(channels),-1)
            values=(d['bolometer'][0]*(d['etendue'][0][None,:]*4*np.pi))*scale
            values=values[:,channels]
            norm=values.max(axis=1)
            if (norm<=0).any(): raise ValueError('invalid acquisition normalization')
            residual=(x[0].reshape(190,-1)@matrix.T-values)/norm[:,None]
            metrics['max_normalized_observation_rmse']=float(np.sqrt(np.mean(residual**2,axis=1)).max())
            outside=~d['vessel_mask'][0].astype(bool)
            metrics['outside_vessel_max']=float(np.max(np.abs(x[0,:,outside])))
        comparisons={'ncc':'>=','nrmse':'<=','max_normalized_observation_rmse':'<=','outside_vessel_max':'<='}
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
