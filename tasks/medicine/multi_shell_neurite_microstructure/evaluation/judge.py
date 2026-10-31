"""Standalone evaluator for every declared microstructure field."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.special import erf

HERE=Path(__file__).resolve().parent
REFERENCE=HERE/'baseline_reference.npz';TRUTH=HERE/'ground_truth.npz';CONFIG=HERE/'metrics.json';RAW=HERE.parent/'data/raw_data.npz'
KEYS=('intra_fraction','intrinsic_diffusivity','extra_transverse_diffusivity','extra_mean_diffusivity')
SHAPE=(1,96,96,19)

def load_prediction(path):
    if not Path(path).is_file():raise ValueError(f'Prediction does not exist: {path}')
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!=set(KEYS):raise ValueError('Prediction must contain exactly all four required field keys')
        arrays={key:f[key] for key in KEYS}
    for key,a in arrays.items():
        if a.shape!=SHAPE:raise ValueError(f'{key}: expected batch-first shape{SHAPE}')
        if a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError(f'{key}: real float32/float64 required')
        if not np.isfinite(a).all():raise ValueError(f'{key}: NaN or Infinity is invalid')
    return arrays

def metrics(estimate,reference):
    p=np.asarray(estimate,dtype=float).ravel();r=np.asarray(reference,dtype=float).ravel()
    span=float(np.ptp(r));norm=float(np.linalg.norm(p)*np.linalg.norm(r))
    if span<=0 or np.linalg.norm(r)==0:raise ValueError('Degenerate evaluator reference')
    ncc=float(p@r/norm) if norm>0 else 0.
    return dict(ncc=ncc,nrmse=float(np.sqrt(np.mean((p-r)**2))/span))

def evaluate(path):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        prediction=load_prediction(path)
        configuration=json.loads(CONFIG.read_text(encoding='utf-8'))
        with np.load(RAW,allow_pickle=False) as f:
            mask=f['observation_mask']; signal=f['diffusion_signal'][mask].astype(float); b=f['bvalues'].ravel()
        if mask.shape!=SHAPE or mask.dtype!=np.bool_:raise ValueError('Malformed observation mask')
        baseline=load_prediction(REFERENCE)
        positive=b>0; unique=np.unique(b[positive]); s0=signal[:,b==0].mean(axis=1)
        if np.any(s0<=0):raise ValueError('Invalid b0 on observation support')
        means=np.stack([signal[:,b==value].mean(axis=1) for value in unique],axis=1)
        counts=np.array([np.sum(b==value) for value in unique])
        def loss(fields):
            v=fields['intra_fraction'][mask,None]; d=fields['intrinsic_diffusivity'][mask,None]
            def average(t):
                root=np.sqrt(np.maximum(unique[None,:]*(d-t),0)); ratio=np.ones_like(root)
                np.divide(np.sqrt(np.pi)*erf(root),2*root,out=ratio,where=root!=0)
                return np.exp(-unique[None,:]*t)*ratio
            predicted=s0[:,None]*(v*average(np.zeros_like(d))+(1-v)*average((1-v)*d))
            return np.sum(counts[None,:]*((predicted-means)/s0[:,None])**2,axis=1)/counts.sum()
        outside=max(float(np.max(np.abs(a[~mask]))) if np.any(~mask) else 0. for a in prediction.values())
        if outside!=0:raise ValueError('Output must be zero outside observation support')
        v=prediction['intra_fraction'][mask]; d=prediction['intrinsic_diffusivity'][mask]
        range_error=max(float(np.max(-v)),float(np.max(v-1)),float(np.max(-d/.00305)),float(np.max(d/.00305-1)),0.)
        derived=max(float(np.max(np.abs(prediction['extra_transverse_diffusivity'][mask]-(1-v)*d))),float(np.max(np.abs(prediction['extra_mean_diffusivity'][mask]-(1-2*v/3)*d))))/.00305
        if range_error>configuration['physical_range_violation_boundary']:
            excess=1e30
        else:excess=float(np.max(loss(prediction)-loss(baseline)))
        measured={'maximum_normalized_mse_excess':excess,'physical_range_violation':range_error,'derived_relative_error':derived,'outside_mask_maximum':outside}
        bounds={key:configuration[key+'_boundary'] for key in measured}
        report['thresholds']={key:dict(comparison='<=',boundary=value) for key,value in bounds.items()}
        if set(bounds)!=set(measured) or not all(np.isfinite(v) and v>=0 for v in bounds.values()):raise ValueError('Invalid evaluator boundaries')
        passed=all(measured[key]<=value for key,value in bounds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured)
        return report,0 if passed else 1
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},errors=[str(exc)])
        return report,2

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True,type=Path);p.add_argument('--report',required=True,type=Path);a=p.parse_args(argv)
    report,code=evaluate(a.prediction)
    try:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:print(f'Cannot write Judge report: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
