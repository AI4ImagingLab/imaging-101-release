"""Independent strict physical-response evaluator; no solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
KEYS=('reconstruction','velocity','acceleration');SHAPE=(1,100001,10)

def load(path):
    with np.load(path,allow_pickle=False) as z:
        if len(z.files)!=3 or set(z.files)!=set(KEYS):raise ValueError('Required exactly reconstruction,velocity,acceleration')
        result={}
        for key in KEYS:
            x=z[key]
            if x.shape!=SHAPE:raise ValueError('Wrong shape for '+key)
            if x.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Expected real float32/float64 '+key)
            if not np.isfinite(x).all():raise ValueError('Nonfinite '+key)
            result[key]=x[0].astype(float)
        return result

def metrics(x,y):
    span=float(np.ptp(y));sx=float(np.max(np.abs(x)));sy=float(np.max(np.abs(y)))
    if not np.isfinite(span) or span<=0 or sy<=0:raise ValueError('Degenerate evaluator reference history')
    a=x/sx if sx else np.zeros_like(x);b=y/sy
    norm=float(np.sqrt(np.sum(a*a)*np.sum(b*b)))
    ncc=float(np.clip(np.sum(a*b)/norm,-1,1)) if norm else 0.
    common=max(sx,sy);rms=float(np.sqrt(np.mean((x/common-y/common)**2)))
    if rms:
        log_error=np.log(rms)+np.log(common)-np.log(span)
        error=float(np.exp(log_error)) if log_error<np.log(np.finfo(float).max) else float(np.finfo(float).max)
    else:error=0.
    return ncc,error

def evaluate(prediction,asset_dir=ROOT):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(asset_dir);config=json.loads((assets/'metrics.json').read_text(encoding='utf-8'))
        reference=assets/'ground_truth.npz'
        if not reference.exists():reference=assets/'baseline_reference.npz'
        target=load(reference);submitted=load(prediction)
        for key in KEYS:
            for floor in range(10):
                ncc,error=metrics(submitted[key][:,floor],target[key][:,floor])
                for metric,value,comparison in [('ncc',ncc,'>='),('nrmse',error,'<=')]:
                    name=f'{key}_floor_{floor+1:02d}_{metric}';bound=float(config[name+'_boundary'])
                    if not np.isfinite(bound) or (metric=='ncc' and not -1<=bound<=1) or (metric=='nrmse' and bound<=0):raise ValueError('Invalid evaluator threshold '+name)
                    report['metrics'][name]=value;report['thresholds'][name]={'comparison':comparison,'boundary':bound}
        report['passed']=all(report['metrics'][n]>=v['boundary'] if v['comparison']=='>=' else report['metrics'][n]<=v['boundary'] for n,v in report['thresholds'].items())
        report['status']='pass' if report['passed'] else 'fail';return report,0 if report['passed'] else 1
    except Exception as e:
        report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(e)]);return report,2
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();report,code=evaluate(a.prediction)
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
