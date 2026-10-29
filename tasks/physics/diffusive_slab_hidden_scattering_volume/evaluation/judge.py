"""Independent, strict NPZ scoring. No solver imports or submitted code execution."""
from pathlib import Path
import argparse,json,sys
import numpy as np

SHAPE=(1,128,32,32)
KEY='reconstruction'

def load_field(path: Path) -> np.ndarray:
    if not path.is_file():raise ValueError(f'Array file unavailable: {path.name}')
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!={KEY}:raise ValueError('Exactly reconstruction key is required')
        x=f[KEY]
    if x.shape!=SHAPE:raise ValueError(f'Expected shape{SHAPE}, got{x.shape}')
    if x.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Expected float32 or float64 real values')
    if not np.isfinite(x).all():raise ValueError('Nonfinite array')
    return x.astype(np.float64)

def measurements(estimate: np.ndarray,reference: np.ndarray) -> dict[str,float]:
    a=estimate.ravel(); b=reference.ravel()
    span=float(b.max()-b.min())
    if not np.isfinite(span) or span<=0:raise ValueError('Reference has undefined dynamic range')
    sa=float(np.max(abs(a))); sb=float(np.max(abs(b)))
    if sb<=0:raise ValueError('Reference has undefined norm')
    aa=a/sa if sa else np.zeros_like(a); bb=b/sb
    cosine=float(np.dot(aa,bb)/(np.linalg.norm(aa)*np.linalg.norm(bb))) if sa else 0.
    scale=max(sa,sb)
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        error=scale*np.sqrt(np.mean((a/scale-b/scale)**2))/span
        maximum=scale*np.max(abs(a/scale-b/scale))/span
    values=dict(ncc=cosine,nrmse=float(error),max_error_range=float(maximum),negative_count=int(np.count_nonzero(a<0)),tail_nonzero_count=int(np.count_nonzero(estimate[:,-15:])))
    if not all(np.isfinite(v) for v in values.values()):raise ValueError('Scoring exceeds finite numeric range')
    return values

def evaluate(prediction: Path,assets: Path | None=None) -> tuple[dict,int]:
    root=Path(__file__).resolve().parent if assets is None else Path(assets)
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((root/'metrics.json').read_text(encoding='utf8'))
        bounds={k:config[k+'_boundary'] for k in ['ncc','nrmse','max_error_range','negative_count','tail_nonzero_count']}
        if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) for v in bounds.values()):raise ValueError('Nonfinite or invalid boundaries')
        if not -1<=bounds['ncc']<=1 or any(bounds[k]<0 for k in bounds if k!='ncc'):raise ValueError('Boundary outside metric domain')
        if bounds['negative_count']!=0 or bounds['tail_nonzero_count']!=0:raise ValueError('Physical count boundaries must be zero')
        ref_path=root/'ground_truth.npz'
        if not ref_path.exists() and not ref_path.is_symlink():ref_path=root/'baseline_reference.npz'
        reference=load_field(ref_path); estimate=load_field(Path(prediction))
        values=measurements(estimate,reference)
        thresholds={k:dict(comparison='>=' if k=='ncc' else '<=',boundary=v) for k,v in bounds.items()}
        passed=all(values[k]>=v if k=='ncc' else values[k]<=v for k,v in bounds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as e:
        report['errors']=[f'{type(e).__name__}: {e}']
        return report,2

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as e:
        print(f'Report-write failure: {e}',file=sys.stderr)
        return 2
    return code

if __name__=='__main__':raise SystemExit(main())
