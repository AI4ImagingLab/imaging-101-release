"""Standalone deterministic NPZ evaluator; no solver imports or execution."""
import argparse,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,4,64,64,10),'dbv':(1,4,64,64,10)}

def load_arrays(path):
    if not Path(path).is_file(): raise ValueError(f'File does not exist: {path}')
    with np.load(path,allow_pickle=False) as archive:
        result={}
        for key,shape in SHAPES.items():
            if key not in archive: raise ValueError(f'Missing required key: {key}')
            value=archive[key]
            if value.shape!=shape: raise ValueError(f'{key}: expected shape {shape}, received {value.shape}')
            if value.dtype!=np.dtype('float64'): raise ValueError(f'{key}: expected real float64, received {value.dtype}')
            if not np.isfinite(value).all(): raise ValueError(f'{key}: non-finite values')
            result[key]=value
    return result

def observation_mask():
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as a: v=a['measurements'][0]
    if v.shape!=(4,64,64,10,24) or not np.isfinite(v).all(): raise ValueError('Invalid public observations')
    x,y,z=np.indices((64,64,10))
    central=(((x-32)/15)**2+((y-32)/18)**2<1)&(z>=2)&(z<=7)
    se=v[...,7];threshold=.5*np.percentile(se,95,axis=(1,2,3))
    return central[None]&(se>threshold[:,None,None,None])&np.all(v>0,axis=-1)

def metrics(prediction,reference):
    mask=observation_mask();result={}
    for c in range(4):
        if not np.any(mask[c]): raise ValueError('Empty evaluation region')
        for name,key in [('r2prime','reconstruction'),('dbv','dbv')]:
            p=prediction[key][0,c];r=reference[key][0,c]
            if np.ptp(r[mask[c]])==0: raise ValueError('Degenerate reference')
            result[f'{name}_rmse_{c}']=float(np.sqrt(np.mean((p[mask[c]]-r[mask[c]])**2)))
        result[f'outside_max_{c}']=float(max(np.max(np.abs(prediction[k][0,c][~mask[c]])) for k in SHAPES))
    return result

def evaluate(prediction_path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        thresholds=cfg['thresholds']
        if set(thresholds)!={f'{name}_{c}' for c in range(4) for name in ('r2prime_rmse','dbv_rmse','outside_max')}: raise ValueError('Invalid metric definitions')
        if any(v['comparison']!='<=' or not np.isfinite(v['boundary']) for k,v in thresholds.items()): raise ValueError('Invalid metric boundary/direction')
        report['thresholds']=thresholds
        refpath=HERE/'ground_truth.npz'
        if not refpath.exists(): refpath=HERE/'baseline_reference.npz'
        ref=load_arrays(refpath);pred=load_arrays(prediction_path)
        measured=metrics(pred,ref)
        if not all(np.isfinite(v) for v in measured.values()): raise ValueError('Non-finite evaluation metric')
        passed=all(measured[k]>=v['boundary'] if v['comparison']=='>=' else measured[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)]
        report['metrics']={}
        return report,2

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:
        print(f'Report write failed: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__': sys.exit(main())
