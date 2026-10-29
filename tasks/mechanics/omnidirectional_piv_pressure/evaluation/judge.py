"""Independent strict NPZ pressure evaluator; no solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
EXPECTED_SHAPE=(1,1001,1001)

def metrics(estimate,reference):
    a=np.asarray(estimate,np.float64).ravel();b=np.asarray(reference,np.float64).ravel()
    span=float(np.ptp(b));norm_b=float(np.linalg.norm(b));norm_a=float(np.linalg.norm(a))
    if span<=0 or norm_b<=0:raise ValueError("Degenerate evaluator reference")
    return {"ncc":float(np.dot(a,b)/(norm_a*norm_b)) if norm_a>0 else 0.0,"nrmse":float(np.sqrt(np.mean((a-b)**2))/span)}

def evaluate(prediction,asset_dir=None):
    report={"status":"error","passed":False,"metrics":{},"thresholds":{},"errors":[]}
    try:
        root=Path(__file__).resolve().parent if asset_dir is None else Path(asset_dir)
        cfg=json.loads((root/"metrics.json").read_text(encoding="utf-8"))
        for k in ("ncc_boundary","nrmse_boundary"):
            if isinstance(cfg[k],bool) or not isinstance(cfg[k],(float,int)) or not np.isfinite(cfg[k]):raise ValueError("Invalid metric configuration")
        if not -1<=cfg["ncc_boundary"]<=1 or cfg["nrmse_boundary"]<0:raise ValueError("Invalid metric boundaries")
        ref_path=root/"ground_truth.npz"
        if not ref_path.exists():ref_path=root/"baseline_reference.npz"
        with np.load(ref_path,allow_pickle=False) as z:ref=z["reconstruction"]
        if ref.shape!=EXPECTED_SHAPE or ref.dtype!=np.float64 or not np.isfinite(ref).all():raise ValueError("Invalid evaluator reference")
        with np.load(Path(prediction),allow_pickle=False) as z:
            if set(z.files)!={"reconstruction"}:raise ValueError("Required output key: reconstruction only")
            p=z["reconstruction"]
        if p.shape!=EXPECTED_SHAPE:raise ValueError(f"Expected shape {EXPECTED_SHAPE}")
        if p.dtype!=np.float64:raise ValueError("Expected real float64 pressure in Pa")
        if not np.isfinite(p).all():raise ValueError("Pressure contains NaN or Infinity")
        if abs(float(p[0,1,1]))>1e-12:raise ValueError("Pressure datum pixel(1,1) must be zero Pa")
        m=metrics(p,ref)
        thresholds={"ncc":{"comparison":">=","boundary":cfg["ncc_boundary"]},"nrmse":{"comparison":"<=","boundary":cfg["nrmse_boundary"]}}
        passed=m["ncc"]>=cfg["ncc_boundary"] and m["nrmse"]<=cfg["nrmse_boundary"]
        report.update(status="pass" if passed else "fail",passed=bool(passed),metrics=m,thresholds=thresholds)
    except Exception as exc:report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prediction",required=True);parser.add_argument("--report",required=True);args=parser.parse_args()
    result=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    except Exception as exc:
        print(f"Cannot write report: {exc}",file=sys.stderr);return 2
    return 0 if result["passed"] else 1 if result["status"]=="fail" else 2

if __name__=="__main__":raise SystemExit(main())
