"""Independent standalone evaluator for all galaxy bands, intrinsic brightness and eight physical parameters."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,3,120,120),'parameters':(1,3,8)}
def load(path):
    if not Path(path).is_file():raise ValueError('Missing NPZ: '+str(path))
    result={}
    with np.load(path,allow_pickle=False)as f:
        for k,shape in SHAPES.items():
            if k not in f:raise ValueError('Missing required key: '+k)
            x=f[k]
            if x.shape!=shape:raise ValueError('Wrong shape for '+k+': expected '+str(shape))
            if x.dtype.kind!='f'or x.dtype.itemsize not in (4,8):raise ValueError('Expected real float32/float64 for '+k)
            if not np.isfinite(x).all():raise ValueError('Nonfinite '+k)
            result[k]=x
    return result
def score(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel();scale=np.ptp(b);norm=np.linalg.norm(a)*np.linalg.norm(b)
    if not np.isfinite(scale)or scale<=0 or np.linalg.norm(b)==0:raise ValueError('Degenerate evaluator reference')
    values=dict(ncc=0. if norm==0 else float(a@b/norm),nrmse=float(np.sqrt(np.mean((a-b)**2))/scale))
    if not all(np.isfinite(v)for v in values.values()):raise ValueError('Metric overflow or nonfinite value')
    return values
def evaluate(prediction):
    r=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        pred=load(prediction);ref_path=HERE/'ground_truth.npz'
        if not ref_path.exists():ref_path=HERE/'baseline_reference.npz'
        ref=load(ref_path);settings=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))
        for channel in range(3):
            for name,value in score(pred['reconstruction'][0,channel],ref['reconstruction'][0,channel]).items():
                boundary=settings[name+'_boundary']
                if not isinstance(boundary,(float,int))or not np.isfinite(boundary):raise ValueError('Invalid boundary')
                label='band_'+str(channel)+'_'+name;r['metrics'][label]=value;r['thresholds'][label]=dict(comparison='>='if name=='ncc'else'<=',boundary=boundary)
            p=pred['parameters'][0,channel];q=ref['parameters'][0,channel]
            tolerance=np.asarray(settings['parameter_absolute_tolerances'],dtype=float).copy()
            if tolerance.shape!=(8,)or not np.isfinite(tolerance).all()or(tolerance<=0).any():raise ValueError('Invalid parameter tolerances')
            fraction=settings['flux_relative_tolerance']
            if not isinstance(fraction,(int,float))or not np.isfinite(fraction)or fraction<=0:raise ValueError('Invalid flux tolerance')
            tolerance[2]=abs(q[2])*fraction
            for k in range(8):
                label='parameter_'+str(channel)+'_'+str(k)+'_absolute_error';r['metrics'][label]=float(abs(p[k]-q[k]));r['thresholds'][label]=dict(comparison='<=',boundary=float(tolerance[k]))
        r['passed']=all((r['metrics'][k]>=t['boundary']if t['comparison']=='>='else r['metrics'][k]<=t['boundary'])for k,t in r['thresholds'].items());r['status']='pass'if r['passed']else'fail';return r,0 if r['passed']else 1
    except Exception as e:r.update(metrics={},passed=False,status='error',errors=[str(e)]);return r,2
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args();report,code=evaluate(args.prediction)
    try:p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as e:print('Report write failure: '+str(e),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
