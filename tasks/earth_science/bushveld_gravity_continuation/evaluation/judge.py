"""Standalone deterministic evaluation; independent of all solver imports."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

SHAPE=(1,48,64)

def score(estimate,reference):
    a=estimate.ravel().astype(np.float64);b=reference.ravel().astype(np.float64)
    anorm=float(np.linalg.norm(a)); bnorm=float(np.linalg.norm(b)); span=float(np.ptp(b))
    if bnorm==0 or span==0:
        raise ValueError('Undefined metrics for zero norm or constant reference')
    ncc=0.0 if anorm==0 else float(np.dot(a,b)/(anorm*bnorm))
    return {'ncc':ncc,'nrmse':float(np.sqrt(np.mean((a-b)**2))/span)}

def validate(path):
    if not Path(path).is_file():raise ValueError('NPZ file does not exist: '+str(path))
    with np.load(path,allow_pickle=False) as archive:
        if 'reconstruction' not in archive.files:raise ValueError('Missing reconstruction key')
        x=archive['reconstruction']
    if x.shape!=SHAPE:raise ValueError('Expected shape '+str(SHAPE))
    if x.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Expected float32 or float64')
    if not np.isfinite(x).all():raise ValueError('Nonfinite reconstruction')
    return x

def evaluate(prediction,asset_dir=None):
    assets=Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        ncc=float(config['ncc_boundary']);nrmse=float(config['nrmse_boundary'])
        if not np.isfinite([ncc,nrmse]).all() or not 0<ncc<=1 or nrmse<=0:raise ValueError('Invalid metric configuration')
        report['thresholds']={'ncc':{'comparison':'>=','boundary':ncc},'nrmse':{'comparison':'<=','boundary':nrmse}}
        reference_path=assets/'ground_truth.npz'
        if not reference_path.exists():reference_path=assets/'baseline_reference.npz'
        try:reference=validate(reference_path)
        except Exception as exc:raise ValueError('Evaluator reference error: '+str(exc)) from exc
        estimate=validate(prediction)
        metrics=score(estimate,reference)
        passed=metrics['ncc']>=ncc and metrics['nrmse']<=nrmse
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)]
        return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        destination=Path(args.report);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print('Report-write failure: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
