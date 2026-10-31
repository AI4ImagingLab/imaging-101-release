"""Independent complete latent-parameter contract; no solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'posterior':(1,63,86),'transition':(1,2,2),'emission':(1,2),'noise':(1,1),'initial':(1,2)}
def load(path):
    with np.load(path,allow_pickle=False) as z:
        arrays={}
        for key,shape in SHAPES.items():
            if key not in z:raise ValueError('Missing key '+key)
            a=z[key]
            if a.shape!=shape or a.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(a).all():raise ValueError('Wrong shape/dtype or nonfinite '+key)
            arrays[key]=a
    return arrays
def score(x,r,lengths):
    active=np.arange(86)[None,:]<lengths[:,None]
    rmse=lambda a:float(np.sqrt(np.mean(a*a)))
    return dict(posterior_rmse=rmse((x['posterior']-r['posterior'])[0,active]),transition_rmse=rmse(x['transition']-r['transition']),initial_rmse=rmse(x['initial']-r['initial']),emission_relative_l2=float(np.linalg.norm(x['emission']-r['emission'])/np.linalg.norm(r['emission'])),noise_relative_error=float(abs(x['noise'][0,0]/r['noise'][0,0]-1)))
def evaluate(path,assets=HERE):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(assets);cfg=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        reference=assets/'ground_truth.npz'
        if not reference.exists():reference=assets/'baseline_reference.npz'
        r=load(reference);x=load(path)
        if np.linalg.norm(r['emission'])<=0 or r['noise'][0,0]<=0:raise ValueError('Degenerate reference')
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as z:lengths=z['lengths'][0]
        metrics=score(x,r,lengths)
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite computed metric')
        limits={k:dict(comparison='<=',boundary=cfg[k+'_boundary']) for k in metrics}
        if not all(np.isfinite(t['boundary']) and t['boundary']>0 for t in limits.values()):raise ValueError('Invalid boundary')
        valid=np.all((x['posterior']>=-1e-6)&(x['posterior']<=1+1e-6)) and np.all(x['noise']>0) and np.all(x['emission']>=0) and x['emission'][0,1]>x['emission'][0,0]
        valid=valid and all(np.all((x[k]>=0)&(x[k]<=1)) for k in ('transition','initial'))
        valid=valid and np.max(abs(x['transition'].sum(axis=1)-1))<=1e-6 and np.max(abs(x['initial'].sum(axis=1)-1))<=1e-6
        active=np.arange(86)[None,:]<lengths[:,None];valid=valid and np.all(x['posterior'][0,~active]==0)
        passed=bool(valid and all(metrics[k]<=v['boundary'] for k,v in limits.items()))
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=limits)
        if not valid:report['errors'].append('Probability, padding, emission-order or positivity contract violated')
    except Exception as e:report.update(metrics={},errors=[str(e)])
    return report
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();r=evaluate(a.prediction)
    try:
        f=Path(a.report);f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[r['status']]
if __name__=='__main__':sys.exit(main())
