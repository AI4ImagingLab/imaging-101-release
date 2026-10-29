"""Independent standalone contract and spectral consistency evaluator."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'weights':(1,6),'reconstruction':(1,160)}
def load(path):
    with np.load(path,allow_pickle=False) as z:
        out={}
        for k,shape in SHAPES.items():
            if k not in z:raise ValueError('Missing key '+k)
            a=z[k]
            if a.shape!=shape:raise ValueError('Invalid shape for '+k)
            if a.dtype.kind!='f' or a.dtype.itemsize not in (4,8):raise ValueError('Expected float32 or float64 dtype for '+k)
            if not np.isfinite(a).all():raise ValueError('Non-finite values in '+k)
            out[k]=a.astype(np.float64)
    if np.any(out['weights']<0) or np.any(out['weights']>1):raise ValueError('Weights outside [0,1]')
    return out
def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        truth=HERE/'ground_truth.npz'
        ref=load(truth if truth.exists() else HERE/'baseline_reference.npz')
        limits=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        names=('weight_max','spectral_rmse','forward_rmse','observation_rmse','simplex_error')
        for n in names:
            b=limits[n+'_boundary']
            if not isinstance(b,(float,int)) or not np.isfinite(b) or b<0:raise ValueError('Invalid metric boundary '+n)
            report['thresholds'][n]={'comparison':'<=','boundary':b}
        p=load(prediction)
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as raw:
            a=raw['unknown'][0];mask=(np.arange(len(a))>=np.searchsorted(a[:,0],11870,side='right')-1)&(np.arange(len(a))<=np.searchsorted(a[:,0],12030,side='right')-1)
            e=a[mask,0];y=a[mask,1]
            s=np.array([np.interp(e,raw[f's{i}'][0,:,0],raw[f's{i}'][0,:,1]) for i in range(1,7)])
        if y.shape!=(160,) or s.shape!=(6,160) or not np.isfinite(s).all() or not np.isfinite(y).all():raise ValueError('Invalid calibration assets')
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            w=p['weights'][0];f=p['reconstruction'][0]
            m=dict(weight_max=float(np.max(abs(w-ref['weights'][0]))),
                   spectral_rmse=float(np.sqrt(np.mean((f-ref['reconstruction'][0])**2))),
                   forward_rmse=float(np.sqrt(np.mean((f-w@s)**2))),
                   observation_rmse=float(np.sqrt(np.mean((f-y)**2))),
                   simplex_error=float(abs(w.sum()-1.)))
        if not all(np.isfinite(v) for v in m.values()):raise ValueError('Non-finite evaluation metric')
        report['metrics']=m;report['passed']=all(m[n]<=limits[n+'_boundary'] for n in names)
        report['status']='pass' if report['passed'] else 'fail'
    except Exception as e:
        report['metrics']={};report['errors']=[str(e)];report['status']='error';report['passed']=False
    return report
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args(argv);r=evaluate(Path(args.prediction))
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:
        print('Report write failed: '+str(e),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[r['status']]
if __name__=='__main__':raise SystemExit(main())
