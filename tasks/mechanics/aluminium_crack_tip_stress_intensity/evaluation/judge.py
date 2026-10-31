"""Independent strict evaluator. No reference-solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,3,231,559),'stress_intensity':(1,2)}

def metrics(x,y):
    # Scaling before arithmetic also gives defined rejection of huge finite submissions.
    sx=float(np.max(np.abs(x)));sy=float(np.max(np.abs(y)))
    if not sy or float(np.ptp(y))==0:raise ValueError('Degenerate evaluator reference')
    a=x/sx if sx else np.zeros_like(x);b=y/sy
    ncc=float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))) if sx else 0.
    scale=max(sx,sy);error=np.sqrt(np.mean((x/scale-y/scale)**2))
    factor=scale/float(np.ptp(y))
    nrmse=float(error*factor)
    if not np.isfinite(nrmse):nrmse=float(np.finfo(np.float64).max)
    return ncc,nrmse

def load(path):
    with np.load(path,allow_pickle=False) as z:
        result={}
        for k,shape in SHAPES.items():
            if k not in z:raise ValueError('Missing required key '+k)
            a=z[k]
            if a.shape!=shape:raise ValueError('Wrong shape for '+k)
            if a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Expected float32/float64 for '+k)
            if not np.isfinite(a).all():raise ValueError('Nonfinite '+k)
            result[k]=a.astype(np.float64)
    return result

def evaluate(prediction,asset_dir=ROOT):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(asset_dir);config=json.loads((assets/'metrics.json').read_text(encoding='utf-8'))
        refpath=assets/'ground_truth.npz'
        if not refpath.exists():refpath=assets/'baseline_reference.npz'
        ref=load(refpath);pred=load(prediction)
        for k in SHAPES:
            ncc,err=metrics(pred[k],ref[k])
            for metric,value,comparison in [('ncc',ncc,'>='),('nrmse',err,'<=')]:
                name=k+'_'+metric;boundary=float(config[name+'_boundary'])
                if not np.isfinite(boundary):raise ValueError('Nonfinite evaluator boundary')
                if metric=='ncc' and not -1<=boundary<=1:raise ValueError('Invalid NCC boundary')
                if metric=='nrmse' and boundary<=0:raise ValueError('Invalid NRMSE boundary')
                report['metrics'][name]=value;report['thresholds'][name]={'comparison':comparison,'boundary':boundary}
        report['passed']=all(report['metrics'][n]>=v['boundary'] if v['comparison']=='>=' else report['metrics'][n]<=v['boundary'] for n,v in report['thresholds'].items())
        report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as e:
        report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(e)])
        return report,2

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    report,code=evaluate(a.prediction)
    try:
        target=Path(a.report);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
