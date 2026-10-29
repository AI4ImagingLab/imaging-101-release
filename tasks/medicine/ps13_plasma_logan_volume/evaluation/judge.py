"""Standalone strict arterial-input Logan volume evaluator; no solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
SHAPE=(1,1,200,200)

def metrics(a,b):
    a=np.asarray(a,np.float64).ravel();b=np.asarray(b,np.float64).ravel();span=float(np.ptp(b));nb=float(np.linalg.norm(b));na=float(np.linalg.norm(a))
    if span<=0 or nb<=0:raise ValueError("Degenerate evaluator reference")
    return {"ncc":float(np.dot(a,b)/(na*nb)) if na else 0.,"nrmse":float(np.sqrt(np.mean((a-b)**2))/span)}

def observation_residual(volume,raw):
    # Independent reconstruction of public calibration/timing conventions.
    mask=raw["valid_mask"][0].astype(bool);a=np.moveaxis(raw["activity"][0],0,-1).astype(np.float64)[mask]
    start=raw["frame_start"][0]/60.;duration=raw["frame_duration"][0]/60.;mid=start+.5*duration
    blood=raw["blood_time"][0]/60.;parent=raw["plasma_activity"][0]*raw["parent_fraction"][0]
    cp=np.interp(mid,np.r_[0.,blood],np.r_[0.,parent])
    integral_a=np.concatenate([np.zeros_like(a[:,:1]),np.cumsum((a[:,1:]+a[:,:-1])*.5*np.diff(mid),axis=-1)],axis=-1)+start[0]*a[:,:1]/2
    integral_p=np.r_[0.,np.cumsum((cp[1:]+cp[:-1])*.5*np.diff(mid))]+start[0]*cp[0]/2
    select=start>=30.;ct=a[:,select]
    if np.any(ct<=0):raise ValueError("Invalid late-time calibration observations")
    x=integral_p[select][None,:]/ct;y=integral_a[:,select]/ct;w=duration[select];v=volume[0][mask]
    offset=np.sum(w*(y-v[:,None]*x),axis=-1)/np.sum(w);residual=y-offset[:,None]-v[:,None]*x
    denom=float(np.sum(w*y*y))
    if denom<=0:raise ValueError("Degenerate transformed observations")
    return float(np.sqrt(np.sum(w*residual*residual)/denom))

def evaluate(prediction,asset_dir=None):
    report={"status":"error","passed":False,"metrics":{},"thresholds":{},"errors":[]}
    try:
        root=Path(__file__).resolve().parent if asset_dir is None else Path(asset_dir)
        cfg=json.loads((root/"metrics.json").read_text(encoding="utf-8"))
        keys=["ncc_boundary","nrmse_boundary","late_linear_relative_weighted_residual_boundary"]
        for k in keys:
            if isinstance(cfg[k],bool) or not isinstance(cfg[k],(float,int)) or not np.isfinite(cfg[k]):raise ValueError("Invalid metric configuration")
        if not -1<=cfg["ncc_boundary"]<=1 or min(cfg[k] for k in keys[1:])<0:raise ValueError("Invalid metric boundaries")
        reference=root/"ground_truth.npz"
        if not reference.exists():reference=root/"baseline_reference.npz"
        with np.load(reference,allow_pickle=False) as z:ref=z["reconstruction"]
        if ref.shape!=SHAPE or ref.dtype!=np.float64 or not np.isfinite(ref).all():raise ValueError("Invalid evaluator reference")
        with np.load(root.parent/"data/raw_data.npz",allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        mask=raw["valid_mask"]
        if mask.shape!=SHAPE or mask.dtype!=np.uint8 or not np.isin(mask,[0,1]).all():raise ValueError("Invalid public evaluation mask")
        mask=mask.astype(bool)
        with np.load(prediction,allow_pickle=False) as z:
            if set(z.files)!={"reconstruction"}:raise ValueError("Required output key: reconstruction only")
            p=z["reconstruction"]
        if p.shape!=SHAPE:raise ValueError(f"Expected shape {SHAPE}")
        if p.dtype!=np.float64:raise ValueError("Expected real float64 distribution volume")
        if not np.isfinite(p).all():raise ValueError("Distribution volume contains NaN or Infinity")
        if np.any(p[~mask]!=0):raise ValueError("Distribution volume must be exactly zero outside valid_mask")
        measured=metrics(p[mask],ref[mask]);measured["late_linear_relative_weighted_residual"]=observation_residual(p,raw)
        thresholds={"ncc":{"comparison":">=","boundary":cfg["ncc_boundary"]},"nrmse":{"comparison":"<=","boundary":cfg["nrmse_boundary"]},"late_linear_relative_weighted_residual":{"comparison":"<=","boundary":cfg["late_linear_relative_weighted_residual_boundary"]}}
        passed=measured["ncc"]>=cfg["ncc_boundary"] and measured["nrmse"]<=cfg["nrmse_boundary"] and measured["late_linear_relative_weighted_residual"]<=cfg["late_linear_relative_weighted_residual_boundary"]
        report.update(status="pass" if passed else "fail",passed=bool(passed),metrics=measured,thresholds=thresholds)
    except Exception as exc:report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prediction",required=True);parser.add_argument("--report",required=True);args=parser.parse_args();result=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    except Exception as exc:print(f"Cannot write report: {exc}",file=sys.stderr);return 2
    return 0 if result["passed"] else 1 if result["status"]=="fail" else 2

if __name__=="__main__":raise SystemExit(main())
