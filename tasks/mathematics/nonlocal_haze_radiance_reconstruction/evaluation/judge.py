"""Independent standalone evaluator; never imports the reconstruction code."""
import argparse,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
CONTRACT={'reconstruction':((1,400,600,3),np.dtype('uint8')),'transmission':((1,400,600),np.dtype('float64'))}

def validate_arrays(path,reference=False):
    if not Path(path).is_file():raise ValueError('missing '+('hidden reference' if reference else 'prediction')+' file')
    with np.load(path,allow_pickle=False) as archive:
        arrays={}
        for key,(shape,dtype) in CONTRACT.items():
            if key not in archive:raise ValueError('missing required key '+key)
            a=archive[key]
            if a.shape!=shape:raise ValueError(key+' shape must be '+str(shape))
            if a.dtype!=dtype:raise ValueError(key+' dtype must be '+str(dtype))
            if not np.isfinite(a).all():raise ValueError(key+' contains non-finite values')
            if key=='transmission' and np.any((a<=0)|(a>1)):raise ValueError('transmission must be in (0,1]')
            arrays[key]=a.astype(np.float64)
    return arrays

def metric_values(estimate,reference):
    measured={}
    for key in CONTRACT:
        x=estimate[key].ravel();r=reference[key].ravel()
        xn=float(np.linalg.norm(x));rn=float(np.linalg.norm(r));span=float(np.ptp(r))
        if rn==0 or span==0:raise ValueError('undefined hidden reference metric for '+key)
        measured[key+'_ncc']=float(np.dot(x,r)/(xn*rn)) if xn else 0.0
        measured[key+'_nrmse']=float(np.sqrt(np.mean((x-r)**2))/span)
    if not all(np.isfinite(v) for v in measured.values()):raise ValueError('non-finite metric')
    return measured

def evaluate(prediction):
    result={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        names=[k+'_'+m for k in CONTRACT for m in ['ncc','nrmse']]
        for name in names:
            value=config[name+'_boundary']
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):raise ValueError('invalid metric boundary '+name)
            if name.endswith('ncc') and not 0<value<=1:raise ValueError('invalid NCC boundary')
            if name.endswith('nrmse') and value<=0:raise ValueError('invalid NRMSE boundary')
            result['thresholds'][name]={'comparison':'>=' if name.endswith('ncc') else '<=','boundary':value}
        truth=HERE/'ground_truth.npz'
        reference=validate_arrays(truth if truth.exists() else HERE/'baseline_reference.npz',True)
        estimate=validate_arrays(prediction)
        values=metric_values(estimate,reference)
        passed=all(v>=config[k+'_boundary'] if k.endswith('ncc') else v<=config[k+'_boundary'] for k,v in values.items())
        result.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values)
        return result,0 if passed else 1
    except Exception as exc:
        result['errors'].append(str(exc))
        return result,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:
        print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
