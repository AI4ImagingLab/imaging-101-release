"""Independent standalone evaluator; imports no solver code."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ASSETS=Path(__file__).resolve().parent
SHAPE=(1,30,30,30)

def score(prediction,reference):
    x,r=prediction.ravel(),reference.ravel()
    spread=float(np.ptp(r));norm=float(np.linalg.norm(x)*np.linalg.norm(r))
    if spread<=0 or np.linalg.norm(r)==0:raise ValueError("Degenerate hidden surface")
    return {"ncc":float(np.dot(x,r)/norm) if norm else 0.0,
            "nrmse":float(np.sqrt(np.mean((x-r)**2))/spread)}

def contact_forward(s,p):
    out=np.full_like(s,-np.inf);h,w=s.shape[-2:];kh,kw=p.shape;oy,ox=(kh-1)//2,(kw-1)//2
    for i in range(kh):
        for j in range(kw):
            dy,dx=i-oy,j-ox;y0,y1=max(0,-dy),min(h,h-dy);x0,x1=max(0,-dx),min(w,w-dx)
            if y0<y1 and x0<x1:out[...,y0:y1,x0:x1]=np.maximum(out[...,y0:y1,x0:x1],s[...,y0+dy:y1+dy,x0+dx:x1+dx]+p[i,j])
    return out

def _load(path,key,shape):
    if not path.is_file():raise ValueError(f"Missing file: {path}")
    with np.load(path,allow_pickle=False) as z:
        if key not in z:raise ValueError(f"Missing required key {key}")
        x=z[key]
    if x.shape!=shape:raise ValueError(f"Expected {key} shape {shape}; got {x.shape}")
    if x.dtype!=np.float64:raise ValueError(f"Expected {key} real float64; got {x.dtype}")
    if not np.isfinite(x).all():raise ValueError(f"Nonfinite values in {key}")
    return x

def evaluate(prediction):
    report={"status":"error","passed":False,"metrics":{},"thresholds":{},"errors":[]}
    try:
        config=json.loads((ASSETS/'metrics.json').read_text(encoding='utf8'))
        threshold={"ncc":{"comparison":">=","boundary":config['ncc_boundary']},"nrmse":{"comparison":"<=","boundary":config['nrmse_boundary']},"observation_rmse":{"comparison":"<=","boundary":config['observation_rmse_boundary']}}
        for name, item in threshold.items():
            boundary=item['boundary']
            if isinstance(boundary,bool) or not isinstance(boundary,(int,float)) or not np.isfinite(boundary):
                raise ValueError(f"Invalid hidden {name} boundary: expected a finite number")
            if (name=='ncc' and not -1<=boundary<=1) or (name!='ncc' and boundary<0):
                raise ValueError(f"Invalid hidden {name} boundary range")
        reference=_load(ASSETS/'ground_truth.npz','reconstruction',SHAPE)
        raw=ASSETS.parent/'data/raw_data.npz';y=_load(raw,'measurements',SHAPE);p=_load(raw,'probe',(1,14,14))[0]
        estimate=_load(Path(prediction),'reconstruction',SHAPE)
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            metrics=score(estimate,reference)
            metrics['observation_rmse']=float(np.sqrt(np.mean((contact_forward(estimate,p)-y)**2)))
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("Nonfinite evaluation metrics")
        passed=metrics['ncc']>=config['ncc_boundary'] and metrics['nrmse']<=config['nrmse_boundary'] and metrics['observation_rmse']<=config['observation_rmse_boundary']
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=threshold)
        return report,0 if passed else 1
    except Exception as e:
        report['errors']=[f"{type(e).__name__}: {e}"]
        return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    result,code=evaluate(args.prediction)
    try:
        dest=Path(args.report);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as e:
        print(f"Cannot write report: {e}",file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
