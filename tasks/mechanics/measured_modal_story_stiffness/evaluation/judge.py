"""Standalone parameter and observed modal-equilibrium evaluator."""
import argparse,json,sys
from pathlib import Path
import numpy as np
SHAPES={'reconstruction':(1,4),'unmeasured_modes':(1,1,2)}
def load(path):
    with np.load(path,allow_pickle=False) as z:
        out={}
        for k,s in SHAPES.items():
            if k not in z:raise ValueError('Missing key '+k)
            a=z[k]
            if a.shape!=s or a.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(a).all():raise ValueError('Wrong shape/dtype or nonfinite '+k)
            out[k]=a
        return out
def dynamic_residual(stiffness,latent,frequency,modes):
    stiffness=np.asarray(stiffness).ravel();u=np.vstack((modes/modes[np.argmax(abs(modes),axis=0),np.arange(2)],np.asarray(latent).reshape(1,2)))
    K=np.zeros((4,4))
    for j,k in enumerate(stiffness):
        K[j,j]+=k
        if j>0:K[j-1,j-1]+=k;K[j,j-1]-=k;K[j-1,j]-=k
    return K@u-u*((2*np.pi*frequency)**2)[None,:]*(12.060/386.088)
def evaluate(prediction,asset_dir=None,data_path=None):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        base=Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent
        conf=json.loads((base/'metrics.json').read_text(encoding='utf8'))
        for name in ['parameter_relative_error','modal_residual_rms']:
            b=conf[name+'_boundary']
            if not np.isfinite(b) or b<=0:raise ValueError('Invalid threshold')
            report['thresholds'][name]={'comparison':'<=','boundary':b}
        p=load(prediction);ref=base/'ground_truth.npz'
        if not ref.exists():ref=base/'baseline_reference.npz'
        r=load(ref)
        if any(np.any(v==0) for v in r.values()):raise ValueError('Zero reference')
        with np.load(data_path if data_path is not None else base.parent/'data/raw_data.npz',allow_pickle=False) as z:f=z['frequency'];v=z['measured_modes']
        if f.shape!=(1,2) or v.shape!=(1,3,2) or not np.isfinite(f).all() or not np.isfinite(v).all() or np.any(np.max(abs(v[0]),axis=0)==0):raise ValueError('Invalid observation assets')
        values=dict(parameter_relative_error=max(float(np.max(abs(p[k]/r[k]-1))) for k in SHAPES),modal_residual_rms=float(np.sqrt(np.mean(dynamic_residual(p['reconstruction'],p['unmeasured_modes'],f[0],v[0])**2))))
        if not all(np.isfinite(x) for x in values.values()):raise ValueError('Undefined metric')
        report['metrics']=values
        valid=np.all((p['reconstruction']>0)&(p['reconstruction']<=20)) and np.all(abs(p['unmeasured_modes'])<=3)
        report['passed']=bool(valid and all(v<=report['thresholds'][k]['boundary'] for k,v in values.items()));report['status']='pass' if report['passed'] else 'fail'
        if not valid:report['errors'].append('Physical parameter bounds violated')
    except Exception as e:report['errors'].append(str(e));report['metrics']={}
    return report
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();r=evaluate(a.prediction)
    try:
        out=Path(a.report);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[r['status']]
if __name__=='__main__':sys.exit(main())
