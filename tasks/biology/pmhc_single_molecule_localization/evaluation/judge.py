"""Independent unordered localization-table evaluator; no solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

def load(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('Missing reconstruction key')
        x=z['reconstruction']
    if x.ndim!=3 or x.shape[0]!=1 or x.shape[2]!=6:raise ValueError('Expected (1,N,6) table')
    if x.dtype!=np.float64:raise ValueError('Expected real float64 dtype')
    if not np.isfinite(x).all():raise ValueError('Nonfinite localization table')
    t=x[0]
    if np.any(t[:,0]!=np.floor(t[:,0])) or np.any(t[:,0]<0) or np.any(t[:,0]>=10):raise ValueError('Frame indices must be integers0..9')
    if np.any(t[:,1:3]<0) or np.any(t[:,1:3]>=128):raise ValueError('Coordinates outside native field')
    if np.any(t[:,3:5]<=0) or np.any(t[:,5]<0):raise ValueError('Mass/size must be positive, eccentricity nonnegative')
    if len(t)>10*128*128:raise ValueError('Too many localizations')
    return t

def score(estimate,reference):
    """Framewise assignment within0.5px; missing references incur unit loss.
    Positional loss is squared Euclidean pixels; mass loss squared relative
    mass error; size/ecc loss squared difference. Every reference contributes;
    false positives additionally lower detection F1. No rescaling/alignment.
    """
    if len(reference)==0:raise ValueError('Empty evaluator reference')
    matched=0;loss=np.zeros(4)
    for f in range(10):
        x=estimate[estimate[:,0]==f];y=reference[reference[:,0]==f]
        m=0;frame_loss=np.zeros(4)
        if len(x) and len(y):
            dist=np.linalg.norm(x[:,None,1:3]-y[None,:,1:3],axis=-1)
            rows,cols=linear_sum_assignment(np.where(dist<=.5,dist,1e6))
            ok=dist[rows,cols]<=.5;rows=rows[ok];cols=cols[ok];m=len(rows)
            frame_loss[0]=np.sum(dist[rows,cols]**2)
            frame_loss[1]=np.sum((x[rows,3]/y[cols,3]-1)**2)
            frame_loss[2]=np.sum((x[rows,4]-y[cols,4])**2)
            frame_loss[3]=np.sum((x[rows,5]-y[cols,5])**2)
        matched+=m;loss+=frame_loss+(len(y)-m)
    rmse=np.sqrt(loss/len(reference))
    return dict(detection_f1=float(2*matched/(len(estimate)+len(reference))),position_rmse_pixels=float(rmse[0]),mass_relative_rmse=float(rmse[1]),size_rmse_pixels=float(rmse[2]),ecc_rmse=float(rmse[3]))

def evaluate(prediction):
    root=Path(__file__).resolve().parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        conf=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
        ref=root/'ground_truth.npz'
        if not ref.exists():ref=root/'baseline_reference.npz'
        reference=load(ref);estimate=load(prediction);metrics=score(estimate,reference)
        thresholds=conf['thresholds']
        if set(metrics)!=set(thresholds):raise ValueError('Metric/threshold mismatch')
        passed=True
        for k,cfg in thresholds.items():
            v=cfg['boundary'];op=cfg['comparison']
            if not np.isfinite(v) or op not in ('<=','>='):raise ValueError('Invalid threshold')
            passed &= metrics[k]<=v if op=='<=' else metrics[k]>=v
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite metrics')
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds)
    except Exception as e:report['errors']=[str(e)]
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();report=evaluate(a.prediction)
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[report['status']]
if __name__=='__main__':sys.exit(main())
