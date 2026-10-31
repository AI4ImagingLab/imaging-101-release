"""Standalone full-energy and Arrhenius observation evaluator."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
def load(path,shape):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('Missing reconstruction')
        a=z['reconstruction']
    if a.shape!=shape or a.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(a).all():raise ValueError('Invalid shape/dtype/finite values')
    return a
def evaluate(path,assets=HERE):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(assets)
        with np.load(assets/'baseline_reference.npz',allow_pickle=False) as z:t=z['observation_times'];beta=z['beta'];start=z['initial_temperature']
        shape=(1,len(t)-1);ref=assets/'ground_truth.npz'
        if not ref.exists():ref=assets/'baseline_reference.npz'
        r=load(ref,shape);p=load(path,shape)
        if not np.isfinite(t).all() or not np.all(np.diff(t,axis=0)>0) or not np.all(r>0):raise ValueError('Invalid hidden assets')
        cfg=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        for k in ['energy_relative_rms','log_integral_rms']:
            b=cfg[k+'_boundary']
            if not np.isfinite(b) or b<=0:raise ValueError('Invalid boundary')
            report['thresholds'][k]=dict(comparison='<=',boundary=b)
        if np.any(p<=0):report.update(status='fail',errors=['Energy must be positive']);return report
        e=p[0,:,None];v=(t[1:]-t[:-1])*(np.exp(-e/(.0083144626*(start+beta*t[:-1])))+np.exp(-e/(.0083144626*(start+beta*t[1:]))))/2
        if np.any(v<=0) or not np.isfinite(v).all():report.update(status='fail',errors=['Undefined Arrhenius integrals']);return report
        logs=np.log(v);logs-=logs.mean(axis=1,keepdims=True)
        m=dict(energy_relative_rms=float(np.linalg.norm(p-r)/np.linalg.norm(r)),log_integral_rms=float(np.sqrt(np.mean(logs**2))))
        report.update(metrics=m,passed=all(v<=report['thresholds'][k]['boundary'] for k,v in m.items()));report['status']='pass' if report['passed'] else 'fail'
    except Exception as e:report.update(errors=[str(e)],metrics={})
    return report
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();r=evaluate(a.prediction)
    try:
        d=Path(a.report);d.parent.mkdir(parents=True,exist_ok=True);d.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[r['status']]
if __name__=='__main__':sys.exit(main())
