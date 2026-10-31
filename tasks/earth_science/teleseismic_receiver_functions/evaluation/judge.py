"""Independent standalone numeric evaluator; never imports solver code."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def score(prediction,reference):
    a=prediction.ravel();b=reference.ravel();den=np.linalg.norm(a)*np.linalg.norm(b)
    if den==0:return dict(ncc=0.,nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))
    return dict(ncc=float(a@b/den),nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))
def load(path):
    if not Path(path).is_file():raise ValueError('Missing NPZ: '+str(path))
    with np.load(path,allow_pickle=False) as f:
        if 'reconstruction' not in f.files:raise ValueError('Missing required reconstruction key')
        x=f['reconstruction']
        if x.shape!=(1,2,4801):raise ValueError('Expected shape (1,2,4801)')
        if x.dtype.kind!='f' or x.dtype.itemsize not in (4,8):raise ValueError('Expected real float32/float64 dtype')
        if not np.isfinite(x).all():raise ValueError('Nonfinite values')
        return x
def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        p=load(prediction); path=HERE/'ground_truth.npz'
        if not path.exists():path=HERE/'baseline_reference.npz'
        r=load(path)
        if np.ptp(r)==0 or np.linalg.norm(r)==0:raise ValueError('Degenerate evaluator reference')
        settings=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))
        thresholds={}
        for component,name in enumerate(('radial','transverse')):
            for metric,value in score(p[:,component],r[:,component]).items():
                key=name+'_'+metric; direction='>=' if metric=='ncc' else '<='
                boundary=settings[metric+'_boundary']
                if not isinstance(boundary,(int,float)) or not np.isfinite(boundary):raise ValueError('Invalid metric boundary')
                thresholds[key]=dict(comparison=direction,boundary=boundary);report['metrics'][key]=value
        report['thresholds']=thresholds
        report['passed']=all((report['metrics'][k]>=v['boundary'] if v['comparison']=='>=' else report['metrics'][k]<=v['boundary'])for k,v in thresholds.items())
        report['status']='pass' if report['passed'] else 'fail';return report,0 if report['passed'] else 1
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},errors=[str(exc)]);return report,2
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prediction',required=True);ap.add_argument('--report',required=True);args=ap.parse_args()
    report,code=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:print('Report write failure: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
