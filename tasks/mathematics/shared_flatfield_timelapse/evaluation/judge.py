"""Standalone physical-quality evaluator; independent of reconstruction code."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.fft import dctn
from scipy.sparse import csr_matrix

NAMES=("objective_counts","mean_abs_residual_counts","flatfield_mean_abs_error")
ROOT=Path(__file__).resolve().parent

def resampling_matrix(n):
    scale=(n-1)/127
    rows=[];cols=[];weights=[]
    for j in range(128):
        center=scale*(j+.5)
        a=max(int(center-scale+.5),0);b=min(int(center+scale+.5),n)
        index=np.arange(a,b)
        w=np.maximum(0,1-np.abs((index-center+.5)/scale))
        w=(w/w.sum()).astype(np.float32)
        rows.extend([j]*len(index));cols.extend(index);weights.extend(w)
    return csr_matrix((np.asarray(weights,dtype=np.float32),(rows,cols)),shape=(128,n))

def thresholds(config):
    if not isinstance(config,dict):raise ValueError("Metric configuration must be an object")
    if {key for key in config if key.endswith('_boundary')}!={name+'_boundary' for name in NAMES}:
        raise ValueError("Metric boundary set must match all three declared quantities")
    if 'comparisons' in config and config['comparisons']!={name:'<=' for name in NAMES}:
        raise ValueError("Metric comparisons must match the declared upper bounds")
    result={}
    for name in NAMES:
        value=config.get(name+"_boundary")
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:
            raise ValueError("Invalid nonnegative finite boundary: "+name)
        result[name]={"comparison":"<=","boundary":float(value)}
    return result

def read_prediction(path):
    if not Path(path).is_file():raise ValueError("Prediction file does not exist")
    with np.load(path,allow_pickle=False) as z:
        result={}
        for key,shape in [("reconstruction",(1,128,128)),("baseline",(1,100))]:
            if key not in z:raise ValueError("Missing required key: "+key)
            a=z[key]
            if a.shape!=shape:raise ValueError("Incorrect shape for "+key)
            if a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError("Expected float32 or float64 array for "+key)
            if not np.all(np.isfinite(a)):raise ValueError("Nonfinite values in "+key)
            result[key]=a[0].astype(np.float64)
    return result

def score(prediction,raw_path,gauge_boundary=1e-5,mae_boundary=1.6162489661058033):
    s=prediction["reconstruction"];b=prediction["baseline"]
    if np.any(s<=0) or np.any(b<=0):
        return {},["Flatfield and temporal baseline must be strictly positive"]
    # Certified public-domain failures before products can overflow. Their
    # positive mean is accumulated with safe scaling of the finite inputs.
    mean_s=float(np.mean(s/s.max())*s.max())
    if abs(mean_s-1)>gauge_boundary:
        return {},["Flatfield violates the public mean-one gauge"]
    # A >= max_t(b_t)*mean(S)/100 - max(I), with max(I)<=255 DN.
    # This only short-circuits an already proven failure of the public MAE.
    if (float(b.max())/100)*mean_s-255>mae_boundary:
        return {},["Temporal background has a certified MAE lower bound above the public boundary"]
    with np.load(raw_path,allow_pickle=False) as z:
        images=z["images"]
        if images.shape!=(1,100,1040,1388) or images.dtype!=np.uint8:raise RuntimeError("Invalid public observation asset")
        images=images[0]
    wy=resampling_matrix(1040);wx=resampling_matrix(1388)
    absolute_error=0.
    for i,frame in enumerate(images):
        coarse=(wx@(wy@frame.astype(np.float32)).T).T
        absolute_error+=np.sum(abs(coarse.astype(np.float64)-b[i]*s))
    mae=float(absolute_error/(100*128**2))
    smooth=float(5/128**2*np.sum(abs(dctn(s*b.mean(),type=2,norm="ortho"))))
    return dict(objective_counts=mae+smooth,mean_abs_residual_counts=mae,
                flatfield_mean_abs_error=float(abs(s.mean()-1))),[]

def evaluate(prediction_path,config_path=None,raw_path=None):
    limits=thresholds(json.loads(Path(config_path or ROOT/"metrics.json").read_text(encoding="utf-8")))
    prediction=read_prediction(prediction_path)
    values,errors=score(prediction,raw_path or ROOT.parent/"data/raw_data.npz",
                        limits['flatfield_mean_abs_error']['boundary'],
                        limits['mean_abs_residual_counts']['boundary'])
    passed=not errors and all(values[n]<=limits[n]["boundary"] for n in NAMES)
    return dict(status="pass" if passed else "fail",passed=passed,metrics=values,thresholds=limits,errors=errors)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prediction",required=True);parser.add_argument("--report",required=True)
    args=parser.parse_args()
    try:report=evaluate(args.prediction);code=0 if report["passed"] else 1
    except Exception as exc:
        report=dict(status="error",passed=False,metrics={},thresholds={},errors=[str(exc)]);code=2
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    except Exception as exc:
        print("Cannot write report: "+str(exc),file=sys.stderr);return 2
    return code

if __name__=="__main__":
    if hasattr(sys.stderr,"reconfigure"):sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
