"""Standalone evaluator; observation PSD is frozen from public trajectory, not truth."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def load(path):
    if not Path(path).is_file():raise ValueError('Missing prediction file')
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('Missing reconstruction key')
        p=z['reconstruction']
    if p.shape!=(1,3):raise ValueError('Expected reconstruction shape (1,3)')
    if p.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Expected float32 or float64')
    if not np.isfinite(p).all():raise ValueError('Prediction must be finite')
    return p
def evaluate(path,assets=ROOT):
    r={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        p=load(path)[0];assets=Path(assets)
        cfg=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        truth=assets/'ground_truth.npz';refpath=truth if truth.exists() else assets/'baseline_reference.npz'
        ref=load(refpath)[0]
        with np.load(assets/'baseline_reference.npz',allow_pickle=False) as z:f=z['frequency'];y=z['density'];a=z['shape']
        if not(np.isfinite(ref).all() and np.all(ref>0) and np.isfinite(y).all() and np.all(y>0) and np.all(a>0) and f.shape==y.shape==a.shape):raise ValueError('Invalid evaluator assets')
        for metric in ('parameter_relative_error','normalized_residual_rms'):
            b=float(cfg[metric+'_boundary'])
            if not np.isfinite(b) or b<=0:raise ValueError('Invalid evaluator threshold')
            r['thresholds'][metric]={'comparison':'<=','boundary':b}
        if np.any(p<=0):
            r.update(status='fail',errors=['Physical parameters must be positive']);return r
        g,k,e=p;tc=g/k
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            model=2*4.1*tc/k*(1+2*tc*400*np.sin(np.pi*f/400)**2*np.sinh(1/(tc*400))/(np.cos(2*np.pi*f/400)-np.cosh(1/(tc*400))))+e*e/400
        if not np.isfinite(model).all():
            r.update(status='fail',errors=['Physical model is not finite']);return r
        r['metrics']={'parameter_relative_error':float(np.max(np.abs(p-ref)/np.abs(ref))),'normalized_residual_rms':float(np.sqrt(np.mean(((y-model)/(y/np.sqrt(a)))**2)))}
        if not all(np.isfinite(v) for v in r['metrics'].values()):raise ValueError('Scoring produced nonfinite metric')
        r['passed']=all(v<=r['thresholds'][k]['boundary'] for k,v in r['metrics'].items());r['status']='pass' if r['passed'] else 'fail'
    except Exception as e:r.update(status='error',passed=False,metrics={},errors=[str(e)])
    return r
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    result=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[result['status']]
if __name__=='__main__':sys.exit(main())
