"""Independent strict joint irradiance/response evaluator; never imports solver."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
CONTRACT={'reconstruction':(1,427,640,3),'response':(1,256,3)}

def read_arrays(path:Path) -> dict:
    if not path.is_file():raise ValueError(f'Missing NPZ: {path}')
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!=set(CONTRACT):raise ValueError('Required keys exactly: reconstruction,response')
        values={k:f[k] for k in CONTRACT}
    for k,v in values.items():
        if v.shape!=CONTRACT[k]:raise ValueError(f'{k}: expected shape{CONTRACT[k]}, got{v.shape}')
        if v.dtype!=np.float64:raise ValueError(f'{k}: float64 required')
        if not np.isfinite(v).all():raise ValueError(f'{k}: nonfinite value')
    return values

def observation_error(irradiance:np.ndarray,response:np.ndarray,raw_path:Path) -> float:
    with np.load(raw_path,allow_pickle=False) as f:frames=f['frames'];times=f['exposure_times']
    if frames.shape!=(1,7,427,640,3) or frames.dtype!=np.uint8 or times.shape!=(1,7) or not np.isfinite(times).all() or np.any(times<=0):
        raise ValueError('Malformed evaluator observation/calibration input')
    error=mass=0.
    for j,t in enumerate(times[0]):
        for c in range(3):
            observed=frames[0,j,...,c].astype(float)
            prediction=np.interp(np.log(irradiance[...,c]*t),np.log(response[:,c]),np.arange(256))
            w=np.minimum(observed,255-observed)*((observed>=8)&(observed<=247))
            error+=np.sum(w*(prediction-observed)**2);mass+=w.sum()
    if mass<=0:raise ValueError('Empty observation evaluation support')
    return float(np.sqrt(error/mass))

def metrics(pred:dict,ref:dict,raw_path:Path) -> dict:
    a=pred['reconstruction'].ravel();b=ref['reconstruction'].ravel()
    return {'ncc':float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),
            'radiance_log_rmse':float(np.sqrt(np.mean(np.log(pred['reconstruction']/ref['reconstruction'])**2))),
            'response_log_rmse':float(np.sqrt(np.mean(np.log(pred['response']/ref['response'])**2))),
            'gauge_max_error':float(np.max(np.abs(pred['response'][0,128]-1))),
            'forward_weighted_adu_rmse':observation_error(pred['reconstruction'][0],pred['response'][0],raw_path)}

def evaluate(prediction:Path,assets:Path=HERE) -> tuple[dict,int]:
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        thresholds={name:{'comparison':op,'boundary':float(config[name+'_boundary'])} for name,op in [('response_log_curvature','<='),('gauge_max_error','<='),('forward_weighted_adu_rmse','<=')]}
        if not all(np.isfinite(x['boundary']) for x in thresholds.values()):raise ValueError('Nonfinite evaluation threshold')
        report['thresholds']=thresholds
        pred=read_arrays(prediction)
        if any(np.any(v<=0) for v in pred.values()) or np.any(np.diff(pred['response'][0],axis=0)<=0):
            report.update(status='fail',errors=['Prediction must be strictly positive with increasing response.']);return report,1
        result={'response_log_curvature':float(np.sum(np.diff(np.log(pred['response'][0]),n=2,axis=0)**2)),'gauge_max_error':float(np.max(abs(pred['response'][0,128]-1))),'forward_weighted_adu_rmse':observation_error(pred['reconstruction'][0],pred['response'][0],assets.parent/'data/raw_data.npz')}
        if not all(np.isfinite(x) for x in result.values()):raise ValueError('Nonfinite metric')
        passed=all(result[k]>=v['boundary'] if v['comparison']=='>=' else result[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=result)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[f'{type(exc).__name__}: {exc}'];return report,2

def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();report,code=evaluate(args.prediction)
    try:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
