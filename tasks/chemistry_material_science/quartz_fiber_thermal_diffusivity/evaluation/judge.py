"""Independent evaluator for the complete thermal parameter submission."""
import argparse,json,sys
from pathlib import Path
import numpy as np
SHAPES={'reconstruction':(1,2,2),'fit_parameters':(1,2,2,2,2),'joint_diffusivity':(1,1)}
def load(path):
    with np.load(path,allow_pickle=False) as z:
        out={}
        for key,shape in SHAPES.items():
            if key not in z:raise ValueError('Missing key: '+key)
            a=z[key]
            if a.shape!=shape or a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Wrong shape or float32/float64 dtype: '+key)
            if not np.isfinite(a).all():raise ValueError('Nonfinite: '+key)
            out[key]=a
    return out
def score(pred,ref,x,observed):
    errors={k:float(np.max(abs(pred[k]/ref[k]-1))) for k in SHAPES}
    c=pred['fit_parameters'][0]
    modeled=c[...,0,None]*x+c[...,1,None]
    rms=np.sqrt(np.mean((modeled-np.moveaxis(observed,-1,-2))**2,axis=-1))
    return dict(relative_error=max(errors['reconstruction'],errors['joint_diffusivity']),coefficient_relative_error=errors['fit_parameters'],log_amplitude_rmse=float(rms[:,:,0].max()),phase_rmse=float(rms[:,:,1].max()))
def evaluate(prediction,asset_dir=None):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        base=Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent
        conf=json.loads((base/'metrics.json').read_text(encoding='utf8'))
        for name in ('relative_error','coefficient_relative_error','log_amplitude_rmse','phase_rmse'):
            b=conf[name+'_boundary']
            if not isinstance(b,(float,int)) or not np.isfinite(b) or b<=0:raise ValueError('Invalid threshold '+name)
            report['thresholds'][name]={'comparison':'<=','boundary':b}
        reference=base/'ground_truth.npz'
        if not reference.exists():reference=base/'baseline_reference.npz'
        p=load(prediction);r=load(reference)
        if any(np.any(v==0) for v in r.values()):raise ValueError('Zero reference in relative metric')
        with np.load(base/'baseline_reference.npz',allow_pickle=False) as z:x=z['observed_distance'][0];y=z['observed_profiles'][0]
        if x.shape!=(308,) or y.shape!=(2,2,308,2) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid observed profiles')
        measured=score(p,r,x,y)
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('Undefined metric')
        report['metrics']=measured;report['passed']=all(v<=report['thresholds'][k]['boundary'] for k,v in measured.items())
        report['status']='pass' if report['passed'] else 'fail'
    except Exception as e:report['errors'].append(str(e));report['metrics']={}
    return report
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    r=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[r['status']]
if __name__=='__main__':sys.exit(main())
