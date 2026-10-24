"""Independent complete-array reference evaluation; never imports solver code."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
SHAPE=(1,101,101,4)
DTYPES={'water':np.dtype('float32'),'fat':np.dtype('float32'),'fat_fraction':np.dtype('float32'),'fieldmap':np.dtype('float64'),'r2star':np.dtype('float64')}
def load_arrays(path):
    with np.load(path,allow_pickle=False) as archive:
        arrays={}
        for key,dtype in DTYPES.items():
            if key not in archive:raise ValueError('missing required key '+key)
            value=archive[key]
            if value.shape!=SHAPE or value.dtype!=dtype or not np.isfinite(value).all():raise ValueError(key+' must be finite '+str(dtype)+' '+str(SHAPE))
            arrays[key]=value
    return arrays
def numeric_metrics(estimate,reference):
    a=np.asarray(estimate,dtype=np.float64).ravel();b=np.asarray(reference,dtype=np.float64).ravel()
    scale=float(np.ptp(b));norm=float(np.linalg.norm(b))
    if scale<=0 or norm<=0:raise ValueError('reference requires positive norm and dynamic range')
    nrmse=float(np.sqrt(np.mean((a-b)**2))/scale)
    norm_a=float(np.linalg.norm(a));ncc=float(np.dot(a,b)/(norm_a*norm)) if norm_a else 0.
    return {'ncc':ncc,'nrmse':nrmse}
def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        estimate=load_arrays(prediction)
        selected=ROOT/'ground_truth.npz' if (ROOT/'ground_truth.npz').exists() else ROOT/'baseline_reference.npz'
        reference=load_arrays(selected)
        config=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'))
        limits={}
        for key in DTYPES:
            for metric,direction in [('ncc','>='),('nrmse','<=')]:
                name=key+'_'+metric;boundary=config[name+'_boundary']
                if isinstance(boundary,bool) or not isinstance(boundary,(int,float)) or not np.isfinite(boundary):raise ValueError('invalid boundary '+name)
                if metric=='ncc' and not -1<=boundary<=1:raise ValueError('NCC boundary outside [-1,1]')
                if metric=='nrmse' and boundary<0:raise ValueError('negative NRMSE boundary')
                limits[name]={'comparison':direction,'boundary':boundary}
        metrics={}
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for key in DTYPES:
                for metric,value in numeric_metrics(estimate[key],reference[key]).items():metrics[key+'_'+metric]=value
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('undefined score')
        passed=all(metrics[k]>=v['boundary'] if v['comparison']=='>=' else metrics[k]<=v['boundary'] for k,v in limits.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=limits)
        return report,0 if passed else 1
    except Exception as error:report['errors']=[type(error).__name__+': '+str(error)];return report,2
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as error:print('Cannot write report: '+str(error),file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
