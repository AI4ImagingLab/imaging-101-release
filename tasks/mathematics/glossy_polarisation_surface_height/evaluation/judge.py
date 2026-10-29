"""Standalone evaluator; no solver imports and no implicit alignment."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

EVALUATION=Path(__file__).resolve().parent
SHAPE=(1,457,235)


def load_array(path, role):
    if not path.is_file():
        raise ValueError(f'{role} NPZ does not exist: {path}')
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'reconstruction'}:
            raise ValueError(f'{role} must contain exactly reconstruction')
        a=z['reconstruction']
    if a.shape!=SHAPE or a.dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise ValueError(f'{role} requires real float32/float64 with shape {SHAPE}')
    if not np.isfinite(a).all():
        raise ValueError(f'{role} contains nonfinite values')
    return a


def evaluate(prediction, evaluation_dir=EVALUATION):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        parameters=json.loads((evaluation_dir/'metrics.json').read_text(encoding='utf-8'))
        boundaries={name:float(parameters[name+'_boundary']) for name in ['ncc','nrmse']}
        if not all(np.isfinite(list(boundaries.values()))) or not 0<boundaries['ncc']<=1 or boundaries['nrmse']<=0:
            raise ValueError('Undefined or invalid evaluator metric boundaries')
        report['thresholds']={
            'ncc':dict(comparison='>=',boundary=boundaries['ncc']),
            'nrmse':dict(comparison='<=',boundary=boundaries['nrmse'])}
        with np.load(evaluation_dir.parent/'data/raw_data.npz',allow_pickle=False) as z:
            mask=z['mask']
        if mask.shape!=SHAPE or mask.dtype!=np.dtype(bool) or not mask.any():
            raise ValueError('Unavailable or malformed public foreground mask')
        reference_path=evaluation_dir/'ground_truth.npz'
        if not reference_path.exists():
            reference_path=evaluation_dir/'baseline_reference.npz'
        ref=load_array(reference_path,'Reference')
        b=ref[mask].astype(np.float64)
        span=float(np.ptp(b));bnorm=float(np.linalg.norm(b))
        if span<=0 or bnorm<=0:
            raise ValueError('Degenerate reference prevents declared metrics')
        a=load_array(Path(prediction),'Prediction')
        if np.any(a[~mask]!=0):
            raise ValueError('Prediction background outside public mask must be zero')
        a=a[mask].astype(np.float64)
        # Normalize before dot products/differences: a finite but extremely
        # poor submission must remain a quality failure with valid JSON.
        ascale=float(np.max(np.abs(a)));bscale=float(np.max(np.abs(b)))
        av=a/ascale if ascale else np.zeros_like(a);bv=b/bscale
        ncc=float(np.clip(np.sum(av*bv)/np.sqrt(np.sum(av*av)*np.sum(bv*bv)),-1.,1.)) if ascale else 0.
        scale=max(ascale,bscale)
        rms_scaled=float(np.sqrt(np.mean((a/scale-b/scale)**2)))
        if rms_scaled==0:nrmse=0.
        else:
            log_error=np.log(rms_scaled)+np.log(scale)-np.log(span)
            nrmse=float(np.exp(log_error)) if log_error<np.log(np.finfo(float).max) else float(np.finfo(float).max)
        measured=dict(ncc=ncc,nrmse=nrmse)
        passed=ncc>=boundaries['ncc'] and nrmse<=boundaries['nrmse']
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)]
        return report,2


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True)
    args=p.parse_args();report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:
        print(f'Cannot write evaluation report: {exc}',file=sys.stderr);return 2
    return code


if __name__=='__main__':
    sys.exit(main())
