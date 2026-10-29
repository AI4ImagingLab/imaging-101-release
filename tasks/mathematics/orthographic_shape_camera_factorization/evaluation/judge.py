"""Standalone joint shape/camera evaluator; no solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,491,3),'cameras':(1,51,2,3),'translations':(1,51,2)}
MAX=float(np.finfo(float).max)

def load(path):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(SHAPES):raise ValueError('Required exactly reconstruction,cameras,translations')
        result={}
        for k,shape in SHAPES.items():
            x=z[k]
            if x.shape!=shape:raise ValueError('Wrong shape for '+k)
            if x.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Expected real float32/float64 '+k)
            if not np.isfinite(x).all():raise ValueError('Nonfinite '+k)
            result[k]=x[0].astype(float)
        return result

def exp_finite(log_value):return MAX if log_value>=np.log(MAX) else float(np.exp(log_value))
def normalized(x):
    scale=float(np.max(np.abs(x)))
    return x/scale if scale else np.zeros_like(x),scale

def compare(normalized_estimate,estimate_scale,reference):
    b,bs=normalized(reference);span=float(np.ptp(reference))
    if not bs or span<=0:raise ValueError('Degenerate evaluator reference')
    a=normalized_estimate
    norm=float(np.sqrt(np.sum(a*a)*np.sum(b*b)))
    ncc=float(np.clip(np.sum(a*b)/norm,-1,1)) if norm else 0.
    common=max(estimate_scale,bs)
    rms=float(np.sqrt(np.mean((a*(estimate_scale/common)-reference/common)**2)))
    error=exp_finite(np.log(rms)+np.log(common)-np.log(span)) if rms else 0.
    return ncc,error

def projection_rmse(shape,cameras,translation,observations):
    x,s=normalized(shape);c,k=normalized(cameras);t,v=normalized(translation);o,w=normalized(observations)
    logp=np.log(s)+np.log(k) if s and k else -np.inf
    logt=np.log(v) if v else -np.inf;logo=np.log(w) if w else -np.inf
    level=max(logp,logt,logo)
    residual=np.einsum('fkc,pc->fpk',c,x)*np.exp(logp-level)+t[:,None,:]*np.exp(logt-level)-o*np.exp(logo-level)
    rms=float(np.sqrt(np.mean(residual*residual)))
    return exp_finite(np.log(rms)+level) if rms else 0.

def evaluate(prediction,asset_dir=ROOT):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(asset_dir);cfg=json.loads((assets/'metrics.json').read_text(encoding='utf-8'))
        reference=assets/'ground_truth.npz'
        if not reference.exists():reference=assets/'baseline_reference.npz'
        ref=load(reference);pred=load(prediction)
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as z:tracks=z['tracks']
        if tracks.shape!=(1,51,491,2) or not np.isfinite(tracks).all():raise ValueError('Unavailable public projection observations')
        x,s=normalized(pred['reconstruction']);y,_=normalized(ref['reconstruction'])
        u,_,vh=np.linalg.svd(x.T@y);gauge=u@vh
        for key in SHAPES:
            a,scale=normalized(pred[key])
            if key in ['reconstruction','cameras']:a=a@gauge
            ncc,error=compare(a,scale,ref[key])
            for metric,value,comparison in [('ncc',ncc,'>='),('nrmse',error,'<=')]:
                name=key+'_'+metric;bound=float(cfg[name+'_boundary'])
                if not np.isfinite(bound) or (metric=='ncc' and not -1<=bound<=1) or (metric=='nrmse' and bound<=0):raise ValueError('Invalid evaluator boundary '+name)
                report['metrics'][name]=value;report['thresholds'][name]={'comparison':comparison,'boundary':bound}
        name='projection_rmse';bound=float(cfg[name+'_boundary'])
        if not np.isfinite(bound) or bound<=0:raise ValueError('Invalid projection boundary')
        report['metrics'][name]=projection_rmse(pred['reconstruction'],pred['cameras'],pred['translations'],tracks[0]);report['thresholds'][name]={'comparison':'<=','boundary':bound}
        report['passed']=all(report['metrics'][n]>=v['boundary'] if v['comparison']=='>=' else report['metrics'][n]<=v['boundary'] for n,v in report['thresholds'].items())
        report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as e:
        report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(e)]);return report,2
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();report,code=evaluate(a.prediction)
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
