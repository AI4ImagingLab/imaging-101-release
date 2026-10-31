"""Independent standalone evaluator for all staggered magnetic components."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'radial':(1,36,180,360),'polar':(1,35,181,360),'azimuthal':(1,35,180,361)}
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
    a=a.ravel();b=b.ravel();scale=np.ptp(b);norm=np.linalg.norm(a)*np.linalg.norm(b)
    if not np.isfinite(scale)or scale<=0 or np.linalg.norm(b)==0:raise ValueError('Degenerate evaluator reference')
    return dict(ncc=0. if norm==0 else float(a@b/norm),nrmse=float(np.sqrt(np.mean((a-b)**2))/scale))
def evaluate(prediction):
    r=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        pred=load(prediction);ref_path=HERE/'ground_truth.npz'
        if not ref_path.exists():ref_path=HERE/'baseline_reference.npz'
        ref=load(ref_path);settings=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))
        for key in SHAPES:
            for name,value in score(pred[key],ref[key]).items():
                boundary=settings[name+'_boundary']
                if not isinstance(boundary,(float,int))or not np.isfinite(boundary):raise ValueError('Invalid boundary')
                label=key+'_'+name;r['metrics'][label]=value;r['thresholds'][label]=dict(comparison='>='if name=='ncc'else'<=',boundary=boundary)
        r['passed']=all((r['metrics'][k]>=t['boundary']if t['comparison']=='>='else r['metrics'][k]<=t['boundary'])for k,t in r['thresholds'].items());r['status']='pass'if r['passed']else'fail';return r,0 if r['passed']else 1
    except Exception as e:r.update(metrics={},passed=False,status='error',errors=[str(e)]);return r,2
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args();report,code=evaluate(args.prediction)
    try:p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as e:print('Report write failure: '+str(e),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
