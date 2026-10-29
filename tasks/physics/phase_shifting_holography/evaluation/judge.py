"""Standalone NPZ Judge. No imports from the solver or submitted code."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ASSETS=Path(__file__).resolve().parent
SHAPE=(1,2,1024,1024)

def score_arrays(estimate: np.ndarray, reference: np.ndarray) -> dict:
    a=estimate.astype(np.float64).ravel(); b=reference.astype(np.float64).ravel()
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    if denom==0:
        ncc=0.0
    else:
        ncc=float(np.dot(a,b)/denom)
    span=float(np.ptp(b))
    if span==0 or np.linalg.norm(b)==0:
        raise ValueError('Evaluator reference is constant or zero norm')
    return dict(ncc_vs_ref=ncc,nrmse_vs_ref=float(np.sqrt(np.mean((a-b)**2))/span))

def evaluate(prediction: str | Path, assets: str | Path=ASSETS) -> tuple[dict,int]:
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(assets)
        config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        boundary=config['gauge_relative_l2_boundary']
        if isinstance(boundary,bool) or not isinstance(boundary,(int,float)) or not np.isfinite(boundary) or boundary<=0:raise ValueError('Invalid evaluator threshold')
        report['thresholds']={'gauge_relative_l2':dict(comparison='<=',boundary=boundary)}
        path=Path(prediction)
        if not path.is_file():
            raise ValueError(f'Prediction file not found: {path}')
        with np.load(path,allow_pickle=False) as data:
            if 'reconstruction' not in data:
                raise ValueError('Missing required key: reconstruction')
            a=data['reconstruction']
        if a.shape!=SHAPE:
            raise ValueError(f'Expected shape {SHAPE}, received {a.shape}')
        if a.dtype not in (np.dtype('float32'),np.dtype('float64')):
            raise ValueError('Expected real float32 or float64 reconstruction')
        if not np.isfinite(a).all():
            raise ValueError('Prediction contains NaN or Infinity')
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as data:
            h=data['holograms']
        if h.shape!=(1,4,1024,1024) or h.dtype!=np.uint8:raise ValueError('Invalid public observations')
        h=h[0].astype(float); h-=h.mean(axis=(-2,-1),keepdims=True)
        q=h[0]+1j*h[1]-h[2]-1j*h[3]
        c=a[0,0].astype(float)+1j*a[0,1].astype(float)
        energy=float(np.sum(np.abs(q)**2))
        if energy<=0:raise ValueError('Degenerate observation quadrature')
        # Correlate cross-products, never divide by zero-valued object pixels.
        z=c*np.conj(q)
        cx=np.sum(z[:,1:]*np.conj(z[:,:-1]));cy=np.sum(z[1:]*np.conj(z[:-1]))
        if abs(cx)==0 or abs(cy)==0:
            error=1.0
        else:
            ax=float(np.angle(cx));ay=float(np.angle(cy))
            yy,xx=np.indices(q.shape);wave=np.exp(1j*(ax*(xx-512)+ay*(yy-512)))
            error=float(np.sqrt(np.sum(np.abs(c-q*wave)**2)/energy))
        measured={'gauge_relative_l2':error}
        if not np.isfinite(error):raise ValueError('Nonfinite scoring arithmetic')
        passed=error<=boundary
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
        return report,2

def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr)
        return 2
    return code

if __name__=='__main__':
    raise SystemExit(main())
