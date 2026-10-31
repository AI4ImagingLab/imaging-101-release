"""Standalone mathematical-criterion evaluator. Never imports solver code."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.fft import dctn
HERE=Path(__file__).resolve().parent
def score(u,coeff,q):
    u=np.asarray(u,dtype=np.float64)
    b=u.reshape(3,54,8,80,8).transpose(0,1,3,2,4)
    c=dctn(b,axes=(-2,-1),norm='ortho',workers=1)
    quant=q[:,None,None].astype(float)
    delta=c-coeff*quant
    dx=np.zeros_like(u);dy=np.zeros_like(u)
    dx[:,:,:-1]=u[:,:,1:]-u[:,:,:-1]
    dy[:,:-1]=u[:,1:]-u[:,:-1]
    variation=np.sqrt(np.sum(dx*dx+dy*dy,axis=0)).sum()/np.sqrt(3)
    fidelity=.001*2*255*np.sqrt(2)/2*np.sum((delta/quant)**2)
    violation=max(0.,float(np.max(np.abs(delta)-.5*quant)))
    return {'objective':float(variation+fidelity),'coefficient_violation':violation}
def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))
        expected={'objective_boundary','coefficient_violation_boundary'}
        if not isinstance(config,dict) or {k for k in config if k.endswith('_boundary')}!=expected:
            raise ValueError('invalid evaluator metric set')
        if config.get('comparisons',{'objective':'<=','coefficient_violation':'<='})!={'objective':'<=','coefficient_violation':'<='}:
            raise ValueError('invalid evaluator comparison directions')
        if any(type(config[k]) not in (int,float) for k in expected):
            raise ValueError('evaluator boundaries must be JSON numbers, excluding booleans')
        thresholds={name:{'comparison':'<=','boundary':config[name+'_boundary']} for name in ['objective','coefficient_violation']}
        if any(not np.isfinite(v['boundary']) or v['boundary']<0 for v in thresholds.values()):raise ValueError('invalid evaluator thresholds')
        report['thresholds']=thresholds
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:
            coeff=z['coefficients'];q=z['quantization']
            if coeff.shape!=(1,3,54,80,8,8) or coeff.dtype!=np.int16:raise ValueError('invalid evaluator coefficient asset')
            if q.shape!=(1,3,8,8) or q.dtype!=np.uint16 or (q==0).any():raise ValueError('invalid evaluator calibration asset')
        path=Path(prediction)
        if not path.is_file():raise ValueError('prediction file does not exist')
        with np.load(path,allow_pickle=False) as z:
            if 'reconstruction' not in z:raise ValueError('missing required key reconstruction')
            u=z['reconstruction']
        if u.shape!=(1,3,432,640):raise ValueError('reconstruction shape must be (1,3,432,640)')
        if u.dtype not in [np.dtype('float32'),np.dtype('float64')]:raise ValueError('reconstruction dtype must be real float32 or float64')
        if not np.isfinite(u).all():raise ValueError('reconstruction contains NaN or Infinity')
        metrics=score(u[0],coeff[0],q[0])
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('nonfinite evaluator metric')
        report['metrics']=metrics
        report['passed']=all(metrics[k]<=v['boundary'] for k,v in thresholds.items())
        report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as error:
        report['errors']=[str(error)];report['metrics']={}
        return report,2
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();result,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as error:
        print('Cannot write judge report: '+str(error),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
