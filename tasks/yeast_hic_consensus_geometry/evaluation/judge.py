"""Independent standalone coordinate and public-contact consistency evaluator."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
REFERENCE=HERE/'baseline_reference.npz'
TRUTH=HERE/'ground_truth.npz'
METRICS=HERE/'metrics.json'
RAW=HERE.parent/'data/raw_data.npz'
SHAPE=(1,350,3)

def load_coordinates(path):
    if not Path(path).is_file():raise ValueError(f'Prediction file does not exist: {path}')
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!={'reconstruction'}:raise ValueError('Expected exactly the reconstruction key')
        x=f['reconstruction']
    if x.shape!=SHAPE:raise ValueError(f'Expected shape {SHAPE}, received {x.shape}')
    if x.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Coordinates must have real float32 or float64 dtype')
    if not np.all(np.isfinite(x)):raise ValueError('Coordinates contain NaN or Infinity')
    return x[0].astype(np.float64,copy=False)

def distance_metrics(prediction,reference,support):
    """All unique supported-locus distances: no crop, scaling or correspondence fit."""
    p=prediction[support];r=reference[support];i,j=np.triu_indices(len(p),1)
    dp=np.sqrt(((p[i]-p[j])**2).sum(1));dr=np.sqrt(((r[i]-r[j])**2).sum(1))
    span=float(dr.max()-dr.min())
    if span<=0:raise ValueError('Degenerate evaluator reference distance range')
    denom=float(np.linalg.norm(dp)*np.linalg.norm(dr))
    ncc=float(np.dot(dp,dr)/denom) if denom>0 else 0.
    return dict(distance_ncc=ncc,distance_nrmse=float(np.sqrt(np.mean((dp-dr)**2))/span))

def contact_objective(x,counts,bias):
    i,j=np.nonzero(counts);d=np.sqrt(((x[i]-x[j])**2).sum(1))
    if np.any(d<=0):return None
    lam=bias[i]*bias[j]*d**-3
    value=float(lam.sum()-(counts[i,j]*np.log(lam)).sum())
    return value if np.isfinite(value) else None

def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads(METRICS.read_text(encoding='utf-8'))
        required={'distance_ncc_boundary':('>=','distance_ncc'),'distance_nrmse_boundary':('<=','distance_nrmse'),'relative_poisson_excess_boundary':('<=','relative_poisson_excess')}
        for name,(comparison,metric) in required.items():
            value=config[name]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:raise ValueError(f'Invalid metric boundary: {name}')
            report['thresholds'][metric]=dict(comparison=comparison,boundary=value)
        if config['distance_ncc_boundary']>1:raise ValueError('Invalid NCC boundary above1')
        reference_path=TRUTH if TRUTH.exists() else REFERENCE
        with np.load(reference_path,allow_pickle=False) as f:
            ref=f['reconstruction'];bias=f['visibility_bias']
        if ref.shape!=SHAPE or ref.dtype!=np.float64 or not np.isfinite(ref).all() or bias.shape!=(350,) or not np.isfinite(bias).all() or np.any(bias<=0):raise ValueError('Malformed evaluator reference')
        with np.load(RAW,allow_pickle=False) as f:counts=f['contact_counts'];mask=f['bead_support']
        if counts.shape!=(1,350,350) or counts.dtype!=np.int64 or mask.shape!=(1,350) or mask.dtype!=np.bool_:raise ValueError('Malformed evaluator public observations')
        mask=mask[0];counts=counts[0]
        if not np.array_equal(mask,counts.sum(0)+counts.sum(1)>0):raise ValueError('Inconsistent observation support')
        if np.any(ref[0,~mask]!=0):raise ValueError('Reference padding is invalid')
        x=load_coordinates(prediction)
        if np.any(x[~mask]!=0):raise ValueError('Unsupported loci must be exactly zero')
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            values=distance_metrics(x,ref[0],mask)
            obj=contact_objective(x,counts,bias);objref=contact_objective(ref[0],counts,bias)
        if objref is None:raise ValueError('Degenerate evaluator reference likelihood')
        # A collapsed but otherwise well-formed prediction is a scientific fail.
        values['relative_poisson_excess']=float((obj-objref)/max(1.,abs(objref))) if obj is not None else 1e100
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Prediction overflows finite distance or contact arithmetic')
        passed=all(values[k]>=v['boundary'] if v['comparison']=='>=' else values[k]<=v['boundary'] for k,v in report['thresholds'].items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)]
        report['metrics']={}
        return report,2

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True,type=Path);p.add_argument('--report',required=True,type=Path);args=p.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:
        print(f'Cannot write judge report: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
