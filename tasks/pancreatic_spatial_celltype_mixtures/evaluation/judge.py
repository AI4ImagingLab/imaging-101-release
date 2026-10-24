"""Independent standalone per-cell-type mixture-field evaluator."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
TRUTH=HERE/"ground_truth.npz"
REFERENCE=HERE/"baseline_reference.npz"
METRICS=HERE/"metrics.json"
CALIBRATION=HERE.parent/"notebooks/reference_outputs/public_preprocessing.npz"
KEYS=("mixing_weights","cell_type_proportions")
SHAPE=(1,428,20)


def load_fields(path):
    if not Path(path).is_file():raise ValueError(f"Missing NPZ: {path}")
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(KEYS):raise ValueError("Expected exactly both declared output keys")
        fields={k:z[k] for k in KEYS}
    for key,x in fields.items():
        if x.shape!=SHAPE:raise ValueError(f"{key}: expected shape {SHAPE}")
        if x.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError(f"{key}: real float32/float64 required")
        if not np.isfinite(x).all():raise ValueError(f"{key}: NaN/Infinity forbidden")
        if np.any(x<0):raise ValueError(f"{key}: negative values violate nonnegative mixture contract")
    return fields


def field_metrics(estimate,reference):
    values={}
    for k in range(20):
        x=estimate[0,:,k].astype(float);r=reference[0,:,k].astype(float)
        span=float(np.ptp(r));norm=float(np.linalg.norm(x)*np.linalg.norm(r))
        if span<=0 or np.linalg.norm(r)==0:raise ValueError(f"Degenerate reference for cell type{k}")
        values[f"type{k:02d}_ncc"]=float(np.dot(x,r)/norm) if norm>0 else 0.
        values[f"type{k:02d}_nrmse"]=float(np.sqrt(np.mean((x-r)**2))/span)
        values[f"type{k:02d}_max_absolute_error"]=float(np.max(abs(x-r)))
    return values


def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        c=json.loads(METRICS.read_text(encoding='utf8'))
        names=['ncc_boundary','nrmse_boundary','max_absolute_error_boundary','mixture_relation_error_boundary','relative_forward_excess_boundary']
        for name in names:
            if isinstance(c[name],bool) or not isinstance(c[name],(int,float)) or not np.isfinite(c[name]) or c[name]<0:raise ValueError(f"Invalid boundary {name}")
        if c['ncc_boundary']>1:raise ValueError("Invalid NCC boundary")
        reference=load_fields(TRUTH if TRUTH.exists() else REFERENCE)
        with np.load(CALIBRATION,allow_pickle=False) as z:U=z['basis'];X=z['observations']
        if U.shape!=(7050,20) or X.shape!=(7050,428) or U.dtype!=np.float64 or X.dtype!=np.float64 or not np.isfinite(U).all() or not np.isfinite(X).all():raise ValueError("Malformed public-input calibration")
        fields=load_fields(prediction);values={};thresholds={}
        for key in KEYS:
            result=field_metrics(fields[key],reference[key])
            for name,value in result.items():
                metric=key+'_'+name;values[metric]=value
                kind='ncc' if name.endswith('_ncc') else 'nrmse' if name.endswith('_nrmse') else 'max_absolute_error'
                thresholds[metric]=dict(comparison='>=' if kind=='ncc' else '<=',boundary=c[kind+'_boundary'])
        V=fields['mixing_weights'][0].astype(float);P=fields['cell_type_proportions'][0].astype(float)
        totals=V.sum(1,keepdims=True)
        values['mixture_relation_error']=float(max(np.max(abs(P.sum(1)-1)),np.max(abs(P-np.divide(V,totals,out=np.zeros_like(V),where=totals>0)))))
        with np.errstate(over='raise',invalid='raise'):
            sse=float(np.sum((X-U@V.T)**2));rse=float(np.sum((X-U@reference['mixing_weights'][0].T)**2))
        if rse<=0:raise ValueError("Degenerate forward reference residual")
        values['relative_forward_excess']=max(0.,(sse-rse)/rse)
        for name in ['mixture_relation_error','relative_forward_excess']:thresholds[name]=dict(comparison='<=',boundary=c[name+'_boundary'])
        if not all(np.isfinite(v) for v in values.values()):raise ValueError("Nonfinite metric arithmetic")
        passed=all(values[k]>=d['boundary'] if d['comparison']=='>=' else values[k]<=d['boundary'] for k,d in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)];report['metrics']={};return report,2


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True,type=Path);p.add_argument('--report',required=True,type=Path);a=p.parse_args(argv)
    report,code=evaluate(a.prediction)
    try:
        a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print(f"Cannot write judge report: {exc}",file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
