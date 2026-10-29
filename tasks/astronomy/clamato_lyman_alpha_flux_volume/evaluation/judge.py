"""Standalone evaluator of the complete real conditional flux map; no solver imports."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ASSET_DIR=Path(__file__).resolve().parent
SHAPE=(1,60,48,876)

def load_field(path):
    with np.load(Path(path),allow_pickle=False) as z:
        if 'reconstruction' not in z:
            raise ValueError('Missing required reconstruction key')
        a=np.array(z['reconstruction'],copy=True)
    if a.shape!=SHAPE:
        raise ValueError(f'reconstruction shape {a.shape}; expected {SHAPE}')
    if a.dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise ValueError('reconstruction must be real float32 or float64')
    if not np.isfinite(a).all():
        raise ValueError('reconstruction contains NaN or Infinity')
    return a

def measure(prediction,reference):
    x=prediction.astype(np.float64,copy=False).ravel()
    y=reference.astype(np.float64,copy=False).ravel()
    span=float(np.ptp(y))
    norm_y=float(np.linalg.norm(y))
    if span<=0 or norm_y<=0:
        raise ValueError('Evaluator reference has undefined dynamic-range metrics')
    norm_x=float(np.linalg.norm(x))
    ncc=float(np.clip(np.dot(x,y)/(norm_x*norm_y),-1,1)) if norm_x>0 else 0.
    nrmse=float(np.sqrt(np.mean((x-y)**2))/span)
    if not np.isfinite([ncc,nrmse]).all():
        raise ValueError('Nonfinite evaluation metric')
    return {'ncc_vs_ref':ncc,'nrmse_vs_ref':nrmse}

def evaluate(prediction_path,asset_dir=None):
    assets=ASSET_DIR if asset_dir is None else Path(asset_dir)
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        nc=float(config['ncc_boundary']);nr=float(config['nrmse_boundary'])
        if not np.isfinite([nc,nr]).all() or not -1<=nc<=1 or nr<0:
            raise ValueError('Invalid evaluator metric boundaries')
        report['thresholds']={'ncc_vs_ref':{'comparison':'>=','boundary':nc},
                              'nrmse_vs_ref':{'comparison':'<=','boundary':nr}}
        truth=assets/'ground_truth.npz'
        reference=load_field(truth if truth.exists() else assets/'baseline_reference.npz')
        if np.ptp(reference)<=0 or np.linalg.norm(reference)==0:
            raise ValueError('Evaluator reference is constant or zero')
    except Exception as e:
        report['errors']=[f'Evaluator assets: {e}'];return report,2
    try:
        prediction=load_field(prediction_path)
        metrics=measure(prediction,reference)
    except Exception as e:
        report['errors']=[f'Invalid prediction: {e}'];return report,2
    passed=metrics['ncc_vs_ref']>=nc and metrics['nrmse_vs_ref']<=nr
    report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics)
    return report,0 if passed else 1

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:
        print(f'Could not write evaluation report: {e}',file=sys.stderr)
        return 2
    print(json.dumps(report,allow_nan=False))
    return code

if __name__=='__main__':
    raise SystemExit(main())
