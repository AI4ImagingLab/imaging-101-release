"""Standalone evaluator for every declared microstructure field."""
from pathlib import Path
import argparse,json,sys
import numpy as np

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
        for key in KEYS:
            for metric,comparison in [('ncc','>='),('nrmse','<=')]:
                boundary=configuration['fields'][key][metric+'_boundary']
                if isinstance(boundary,bool) or not isinstance(boundary,(int,float)) or not np.isfinite(boundary) or boundary<0 or (metric=='ncc' and boundary>1):raise ValueError('Invalid evaluator metric configuration')
                report['thresholds'][key+'.'+metric]=dict(comparison=comparison,boundary=boundary)
        reference_path=TRUTH if TRUTH.exists() else REFERENCE
        with np.load(reference_path,allow_pickle=False) as f:
            if set(f.files)!=set(KEYS):raise ValueError('Malformed evaluator reference keys')
            reference={key:f[key] for key in KEYS}
        with np.load(RAW,allow_pickle=False) as f:mask=f['observation_mask']
        if mask.shape!=SHAPE or mask.dtype!=np.bool_ or mask.sum()!=44427:raise ValueError('Malformed evaluator fixed observation support')
        measured={}
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for key in KEYS:
                r=reference[key];p=prediction[key]
                if r.shape!=SHAPE or r.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(r).all() or np.any(r[~mask]!=0):raise ValueError('Malformed evaluator reference field')
                if np.any(p[~mask]!=0):raise ValueError(f'{key}: output must be zero outside the fixed observation mask')
                for metric,value in metrics(p[mask],r[mask]).items():measured[key+'.'+metric]=value
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('Prediction produces nonfinite scoring arithmetic')
        passed=all(measured[key]>=t['boundary'] if t['comparison']=='>=' else measured[key]<=t['boundary'] for key,t in report['thresholds'].items())
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
