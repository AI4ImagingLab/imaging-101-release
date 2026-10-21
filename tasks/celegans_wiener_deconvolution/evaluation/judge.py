"""Standalone deterministic scorer; never imports the reconstruction code."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent

def read_array(path: Path) -> np.ndarray:
    if not path.is_file(): raise ValueError(f'File not found: {path}')
    with np.load(path,allow_pickle=False) as data:
        if 'reconstruction' not in data: raise ValueError('Missing reconstruction key')
        array=data['reconstruction']
    if array.shape!=(1,310,310): raise ValueError('Expected reconstruction shape (1,310,310)')
    if array.dtype not in (np.dtype('float32'),np.dtype('float64')): raise ValueError('Expected real float32 or float64 dtype')
    if not np.isfinite(array).all(): raise ValueError('Nonfinite reconstruction')
    return array.astype(np.float64)

def score(estimate: np.ndarray,reference: np.ndarray) -> dict[str,float]:
    a,b=estimate.ravel(),reference.ravel()
    dynamic=float(np.ptp(b));nb=float(np.linalg.norm(b));na=float(np.linalg.norm(a))
    if dynamic<=0 or nb<=0: raise ValueError('Degenerate evaluator reference')
    return {'ncc':float(np.dot(a,b)/(na*nb)) if na>0 else 0.0,
            'nrmse':float(np.sqrt(np.mean((a-b)**2))/dynamic)}

def evaluate(prediction: Path) -> tuple[dict,int]:
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))
        thresholds={'ncc':{'comparison':'>=','boundary':float(config['ncc_boundary'])},
                    'nrmse':{'comparison':'<=','boundary':float(config['nrmse_boundary'])}}
        if not all(np.isfinite(v['boundary']) for v in thresholds.values()): raise ValueError('Nonfinite metric boundary')
        if not 0<config['ncc_boundary']<=1 or config['nrmse_boundary']<=0: raise ValueError('Invalid metric boundaries')
        report['thresholds']=thresholds
        truth=HERE/'ground_truth.npz'
        reference=read_array(truth if truth.exists() else HERE/'baseline_reference.npz')
        estimate=read_array(prediction)
        metrics=score(estimate,reference)
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite metric')
        passed=metrics['ncc']>=config['ncc_boundary'] and metrics['nrmse']<=config['nrmse_boundary']
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)]
        return report,2

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',required=True,type=Path)
    parser.add_argument('--report',required=True,type=Path)
    args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
