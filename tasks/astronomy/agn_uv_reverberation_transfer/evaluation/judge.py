"""Independent standalone evaluator, never imports solver modules."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,3,600),'transfer':(1,2,3),'continuum':(1,2)}
def load(path):
    with np.load(path,allow_pickle=False) as z:
        out={}
        for key,shape in SHAPES.items():
            if key not in z:raise ValueError('Missing key '+key)
            a=z[key]
            if a.shape!=shape:raise ValueError('Wrong shape '+key)
            if a.dtype.kind!='f' or a.dtype.itemsize not in (4,8):raise ValueError('Expected real float32/float64 '+key)
            if not np.isfinite(a).all():raise ValueError('Nonfinite '+key)
            out[key]=a
    if np.any(out['continuum']<=0) or np.any(out['transfer']<0) or np.any(out['transfer'][:,:,1:]<=0):raise ValueError('Invalid physical parameter')
    return out

def evaluate(path):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        cfg=json.loads((HERE/'metrics.json').read_text());refpath=HERE/'ground_truth.npz'
        ref=load(refpath if refpath.exists() else HERE/'baseline_reference.npz');pred=load(path)
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:obs=z['observations'][0]
        if obs.shape!=(3,60,3) or not np.isfinite(obs).all() or np.any(obs[:,:,2]<=0):raise ValueError('Invalid public observation asset')
        start=obs[:,:,0].min();end=obs[:,:,0].max();times=np.linspace(start-.1*(end-start),end+.1*(end-start),600)
        m=dict(transfer_scaled_error=float(np.max(abs(pred['transfer']-ref['transfer'])/np.array([10,10,1]))),continuum_log_error=float(np.max(abs(np.log(pred['continuum']/ref['continuum'])))))
        bounds={k:cfg[k+'_boundary'] for k in m}
        for i in range(3):
            m['profile_rmse_'+str(i)]=float(np.sqrt(np.mean((pred['reconstruction'][0,i]-ref['reconstruction'][0,i])**2)))
            bounds['profile_rmse_'+str(i)]=cfg['profile_rmse_boundary']
            m['observation_wrms_'+str(i)]=float(np.sqrt(np.mean(((np.interp(obs[i,:,0],times,pred['reconstruction'][0,i])-obs[i,:,1])/obs[i,:,2])**2)))
            bounds['observation_wrms_'+str(i)]=cfg['observation_wrms_boundary'][i]
        if not all(np.isfinite(v) and v>0 for v in bounds.values()):raise ValueError('Invalid metric boundary')
        if not all(np.isfinite(v) for v in m.values()):raise ValueError('Undefined metric')
        report['thresholds']={k:dict(comparison='<=',boundary=v) for k,v in bounds.items()};report['metrics']=m
        report['passed']=all(m[k]<=v for k,v in bounds.items());report['status']='pass' if report['passed'] else 'fail'
    except Exception as e:report.update(status='error',passed=False,metrics={},errors=[str(e)])
    return report

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args(argv);r=evaluate(a.prediction)
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[r['status']]
if __name__=='__main__':raise SystemExit(main())
