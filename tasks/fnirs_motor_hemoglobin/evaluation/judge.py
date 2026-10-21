"""Standalone evaluator. Does not import reference or submitted solver code."""
import argparse,json,sys
from pathlib import Path
import numpy as np

SHAPE=(1,68,2,4860)

def load(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z: raise ValueError('Missing reconstruction key')
        x=z['reconstruction']
    if x.shape!=SHAPE: raise ValueError(f'Expected shape {SHAPE}, got {x.shape}')
    if x.dtype!=np.float64: raise ValueError('Expected real float64 dtype')
    if not np.isfinite(x).all(): raise ValueError('Nonfinite reconstruction')
    return x

def component_metrics(x,y):
    """Cosine NCC (no centering), dynamic-range normalized RMS error."""
    dx=float(np.ptp(y)); nx=float(np.linalg.norm(x)); ny=float(np.linalg.norm(y))
    if dx<=0 or ny<=0: raise ValueError('Undefined reference metrics')
    return {'ncc':float(np.sum(x*y)/(nx*ny)) if nx>0 else 0.0,
            'nrmse':float(np.sqrt(np.mean((x-y)**2))/dx)}

def evaluate(prediction):
    root=Path(__file__).resolve().parent
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
        ref=root/'ground_truth.npz'
        if not ref.exists(): ref=root/'baseline_reference.npz'
        reference=load(ref);x=load(prediction)
        metrics={}
        for i,name in enumerate(('hbo','hbr')):
            for k,v in component_metrics(x[:,:,i,:],reference[:,:,i,:]).items():metrics[name+'_'+k]=v
        with np.load(root.parent/'data/raw_data.npz',allow_pickle=False) as z:
            intensity=z['intensity']; e=z['extinction'];d=z['distances']
        od=-np.log(intensity/intensity.mean(axis=-1,keepdims=True))
        # Independent Einstein contraction of E*d*PPF*1000, M->OD.
        pred=np.einsum('bij,bp,bpjt->bpit',e,d,x)*6000.
        metrics['forward_relative_rms']=float(np.linalg.norm(pred-od)/np.linalg.norm(od))
        thresholds=config['thresholds']
        if set(thresholds)!=set(metrics):raise ValueError('Metric/threshold mismatch')
        passed=True
        for k,v in thresholds.items():
            boundary=v['boundary']; comparison=v['comparison']
            if not np.isfinite(boundary) or comparison not in ('>=','<='):raise ValueError('Invalid threshold configuration')
            passed &= metrics[k]>=boundary if comparison=='>=' else metrics[k]<=boundary
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite metric')
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds)
    except Exception as exc:report['errors']=[str(exc)]
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    result=evaluate(a.prediction)
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[result['status']]

if __name__=='__main__':sys.exit(main())
