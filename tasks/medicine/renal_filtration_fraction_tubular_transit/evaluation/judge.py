"""Standalone deterministic NPZ evaluator; no solver imports or execution."""
import argparse,json,sys,importlib.util
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('renal_forward_contract',HERE/'forward_contract.py');_forward=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_forward)
SHAPES={'reconstruction':(1,228,4),'signal':(1,228,125)}

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

def metrics(prediction,reference):
    p=prediction['reconstruction'][0];q=reference['reconstruction'][0]
    if np.any(p[:,3]<=0) or np.any(q[:,3]<=0):raise ValueError('Transit times must be positive')
    result={}
    for i,(name,scale) in enumerate([('flow',.05),('volume',.3),('filtration',.3)]):
        result[name+'_scaled_rmse']=float(np.sqrt(np.mean(((p[:,i]-q[:,i])/scale)**2)))
    result['transit_log_rmse']=float(np.sqrt(np.mean(np.log(p[:,3]/q[:,3])**2)))
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as a:
        lengths=a['lengths'][0];observed=a['measurements'][0]
    ref_error=[];obs_error=[];padding=[]
    for i,L in enumerate(lengths):
        s=prediction['signal'][0,i,:L];r=reference['signal'][0,i,:L];y=observed[i,:L]
        if np.linalg.norm(r)==0 or np.linalg.norm(y)==0:raise ValueError('Degenerate signal')
        ref_error.append(np.linalg.norm(s-r)/np.linalg.norm(r))
        obs_error.append(np.linalg.norm(s-y)/np.linalg.norm(y))
        padding.extend(np.abs(prediction['signal'][0,i,L:]))
    result['signal_reference_max_relative_rmse']=float(max(ref_error))
    result['signal_observation_max_relative_rmse']=float(max(obs_error))
    result['padding_max']=float(max(padding,default=0.))
    return result

def evaluate(prediction_path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        thresholds=cfg['thresholds']
        if set(thresholds)!={'signal_model_identity_max_relative_rmse','maximum_fit_quality_budget_ratio','signal_observation_max_relative_rmse','padding_max'}: raise ValueError('Invalid metric definitions')
        if any(v['comparison']!='<=' or not np.isfinite(v['boundary']) for k,v in thresholds.items()): raise ValueError('Invalid metric boundary/direction')
        report['thresholds']=thresholds
        pred=load_arrays(prediction_path)
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
        budgets=np.asarray(cfg['fit_quality_budgets'])
        if budgets.shape!=(228,) or not np.isfinite(budgets).all() or np.any(budgets<=0):raise ValueError('Invalid fit-quality budgets')
        measured=_forward.physical_metrics(pred,data,budgets)
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
