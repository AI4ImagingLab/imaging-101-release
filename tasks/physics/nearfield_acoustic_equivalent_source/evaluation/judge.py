"""Standalone NumPy evaluator; never imports solver or upstream modules."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ASSETS=Path(__file__).resolve().parent
SHAPE=(1,40,50)
def load(path,key='reconstruction',shape=SHAPE):
    with np.load(path,allow_pickle=False) as z:
        if key not in z:raise ValueError('Missing required key '+key)
        a=z[key]
    if a.shape!=shape or a.dtype!=np.complex64:raise ValueError(f'{key} must be complex64 {shape}; received {a.dtype} {a.shape}')
    if not np.isfinite(a).all():raise ValueError('Nonfinite '+key)
    return a
def score(x,r):
    a=x.astype(np.complex128).ravel();b=r.astype(np.complex128).ravel()
    norm=np.linalg.norm(b)
    if norm<=0:raise ValueError('Degenerate reference')
    amag=np.abs(a);bmag=np.abs(b);den=np.linalg.norm(amag)*np.linalg.norm(bmag)
    return {'relative_complex_l2':float(np.linalg.norm(a-b)/norm),'magnitude_cosine':float(np.dot(amag,bmag)/den) if den else 0.}
def acquisition():
    pressure=load(ASSETS.parent/'data/raw_data.npz','pressure',(1,9,12))[0].astype(np.complex128).ravel()
    sx,sy=np.meshgrid(np.linspace(0.,.825,50),np.linspace(0.,.6,40))
    mx,my=np.meshgrid(np.linspace(0.,.825,12),np.linspace(0.,.6,9))
    radius=np.sqrt((mx.ravel()[:,None]-sx.ravel()[None,:])**2+(my.ravel()[:,None]-sy.ravel()[None,:])**2+.025**2)
    omega=2*np.pi*1150.;G=1j*omega*1.2*np.exp(-1j*omega/343.*radius)/(4*np.pi*radius)
    return G,pressure

def objective_metrics(q):
    G,pressure=acquisition();a=q[0].astype(np.complex128)
    delta=G@a.ravel()-pressure
    dx=np.diff(np.pad(a,((0,0),(0,2))),n=2,axis=1)
    dy=np.diff(np.pad(a,((0,2),(0,0))),n=2,axis=0)
    weight=np.linalg.norm(G,2)*10**(-35/20)*np.linalg.norm(pressure)/np.sqrt(a.size)
    objective=float(np.vdot(delta,delta).real+weight*np.sum(np.sqrt(abs(dx)**2+abs(dy)**2)))
    norm=np.linalg.norm(pressure)
    if norm<=0:raise ValueError('Degenerate observation')
    return {'regularized_objective':objective,'observation_relative_l2':float(np.linalg.norm(delta)/norm)}

def observation_residual(q):return objective_metrics(q)['observation_relative_l2']
def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((ASSETS/'metrics.json').read_text(encoding='utf-8'))
        names=['regularized_objective','observation_relative_l2'];limits={k:cfg[k+'_boundary'] for k in names}
        for key,value in limits.items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):raise ValueError('Invalid boundary '+key)
            if value<0 or (key=='magnitude_cosine' and value>1):raise ValueError('Invalid boundary range '+key)
        x=load(Path(prediction))
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            metrics=objective_metrics(x)
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite metric')
        passed=all(metrics[k]<=limits[k] for k in names)
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds={k:{'comparison':'>=' if k=='magnitude_cosine' else '<=','boundary':v} for k,v in limits.items()})
        return report,0 if passed else 1
    except Exception as exc:report['errors']=[f'{type(exc).__name__}: {exc}'];return report,2
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args(argv)
    report,code=evaluate(a.prediction)
    try:
        target=Path(a.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
