"""Standalone physical elastic-lidar consistency and full-field evaluator."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
METRICS=('backscatter_ncc','backscatter_nrmse','extinction_ncc','extinction_nrmse','backscatter_return_rms','extinction_return_rms')

def read_output(path):
    with np.load(path,allow_pickle=False) as z:out={k:z[k].copy() for k in ('reconstruction','extinction')}
    for k,a in out.items():
        if a.shape!=(1,66,759) or a.dtype!=np.dtype('float64') or np.iscomplexobj(a) or not np.isfinite(a).all():raise ValueError(k+' must be finite real float64(1,66,759)')
    return out

def compare(a,b):
    x=a.ravel();y=b.ravel();dynamic=np.ptp(y)
    if not dynamic>0 or not np.linalg.norm(y)>0:raise ValueError('degenerate evaluator reference')
    scale=max(np.max(abs(x)),1.);xx=x/scale;norm=np.linalg.norm(xx)
    cosine=0. if norm==0 else float(xx@y/norm/np.linalg.norm(y))
    return cosine,float(np.sqrt(np.mean(((x-y)/dynamic)**2)))

def return_rms(beta,signal,molecular_beta,molecular_alpha,step=7.5,lidar_ratio=50.):
    """Independent cumulative optical-depth forward model on all evaluated bins."""
    total=beta+molecular_beta
    if np.any(total<=0):raise ValueError('nonpositive total atmospheric backscatter')
    alpha=lidar_ratio*beta+molecular_alpha
    prefix=np.concatenate([np.zeros((len(beta),1)),np.cumsum((alpha[:,:-1]+alpha[:,1:])*(step/2),axis=1)],axis=1)
    from_reference=prefix-prefix[:,-1,None]
    prediction=total/total[:,-1,None]*np.exp(-2*from_reference)
    observed=signal/signal[:,-1,None]
    return float(np.sqrt(np.mean((prediction-observed)**2))/np.sqrt(np.mean(observed**2)))

def evaluate(prediction):
    config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'));thresholds={}
    for key in METRICS:
        boundary=config[key+'_boundary'];cosine=key.endswith('_ncc')
        if isinstance(boundary,bool) or not isinstance(boundary,(int,float)) or not np.isfinite(boundary) or (cosine and not -1<=boundary<=1) or (not cosine and boundary<0):raise ValueError('invalid metric boundary '+key)
        thresholds[key]={'comparison':'>=' if cosine else '<=','boundary':boundary}
    estimate=read_output(prediction);reference=HERE/'ground_truth.npz'
    if not reference.exists():reference=HERE/'baseline_reference.npz'
    reference=read_output(reference);values={}
    for key,name in [('reconstruction','backscatter'),('extinction','extinction')]:values[name+'_ncc'],values[name+'_nrmse']=compare(estimate[key],reference[key])
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:
        signal=z['range_corrected_signal'][0,:,54:813];bm=z['molecular_backscatter'][0,54:813];am=z['molecular_extinction'][0,54:813]
    values['backscatter_return_rms']=return_rms(estimate['reconstruction'][0],signal,bm,am)
    values['extinction_return_rms']=return_rms(estimate['extinction'][0]/50.,signal,bm,am)
    if not np.isfinite(list(values.values())).all():raise ValueError('nonfinite computed metrics')
    passed=all(values[k]>=v['boundary'] if v['comparison']=='>=' else values[k]<=v['boundary'] for k,v in thresholds.items())
    return {'status':'pass' if passed else 'fail','passed':passed,'metrics':values,'thresholds':thresholds,'errors':[]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    try:report=evaluate(args.prediction);code=0 if report['passed'] else 1
    except Exception as exc:report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[str(exc)]};code=2
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('report write failed: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
