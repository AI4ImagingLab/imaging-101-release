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


def profile_assets():
    # Frozen expression arrays are independently derived only from public counts;
    # provenance and raw/preprocessing parity are covered by comparing tests.
    with np.load(CALIBRATION,allow_pickle=False) as z: U=z['basis'];X=z['observations']
    if U.shape!=(7050,20) or X.shape!=(7050,428) or not np.isfinite(U).all() or not np.isfinite(X).all():raise ValueError('Invalid count-derived expression calibration')
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:xy=z['spatial_coordinates'][0].copy()
    xy-=xy.min(0);xy/=xy.max();delta=xy[:,None,:]-xy[None,:,:]
    W=np.exp(-np.sum(delta*delta,axis=2)/.02);np.fill_diagonal(W,0)
    assets=[]
    for phi in (.01,.1,.3,.5,.7,.9,.99):
        L=np.diag(W.sum(1))-phi*W;sign,ld=np.linalg.slogdet(L)
        if sign!=1:raise ValueError('Invalid CAR precision')
        assets.append((L,10*ld))
    return U,X,assets


def profile_score(V,assets):
    U,X,precisions=assets;n,k=V.shape
    # Direct residuals avoid cancellation in the source trace expression.
    residual=X-U@V.T;s=float(np.mean(residual*residual))
    if not np.isfinite(s) or s<=0:raise ValueError('Degenerate Gaussian residual variance')
    best=-np.inf
    for L,normalizer in precisions:
        b=np.sum(L@V,axis=0)/L.sum();Z=V-b
        q=np.sum(Z*(L@Z),axis=0);lam=(q+n)/(n+4)
        if np.any(lam<=0):raise ValueError('Invalid profiled CAR variance')
        score=-X.size*.5*(np.log(s)+1)-(n*.5+2)*np.log(lam).sum()-.5*np.sum((q+n)/lam)+normalizer
        best=max(best,float(score))
    return best/X.size


def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        c=json.loads(METRICS.read_text(encoding='utf8'))
        names=['profile_nll_excess','mixture_relation_error']
        for name in names:
            v=c[name+'_boundary']
            if type(v) not in (float,int) or not np.isfinite(v) or v<0:raise ValueError('Invalid boundary '+name)
        if TRUTH.exists():raise ValueError('Unexpected actual truth: v2 conditional-profile protocol requires explicit truth metric certification')
        reference=load_fields(REFERENCE);fields=load_fields(prediction)
        for role,f in [('Reference',reference),('Prediction',fields)]:
            if np.any(f['mixing_weights'].sum(2)<=0):raise ValueError(role+' has zero mixture rows')
        assets=profile_assets();V=fields['mixing_weights'][0].astype(float);P=fields['cell_type_proportions'][0].astype(float)
        reference_V=reference['mixing_weights'][0].astype(float)
        if np.max(abs(reference['cell_type_proportions'][0]-reference_V/reference_V.sum(1,keepdims=True)))>2e-6:raise ValueError('Inconsistent evaluator proportions')
        with np.errstate(over='raise',invalid='raise'):
            values={'profile_nll_excess':profile_score(reference_V,assets)-profile_score(V,assets),
                    'mixture_relation_error':float(max(np.max(abs(P.sum(1)-1)),np.max(abs(P-V/V.sum(1,keepdims=True)))))}
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Nonfinite profile metric')
        thresholds={n:dict(comparison='<=',boundary=c[n+'_boundary']) for n in names}
        passed=all(values[n]<=thresholds[n]['boundary'] for n in names)
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
