"""Standalone evaluation of complete density and conditional uncertainty fields."""
from pathlib import Path
import argparse,json,sys
import numpy as np

KEYS=('reconstruction','density_sigma')
def arrays(path):
    if not Path(path).is_file():raise ValueError('Missing NPZ: '+str(path))
    result={}
    with np.load(path,allow_pickle=False) as f:
        for key in KEYS:
            if key not in f:raise ValueError('Missing required key '+key)
            x=f[key]
            if x.shape!=(1,32,32) or x.dtype!=np.dtype('float64'):raise ValueError('Expected float64(1,32,32) for '+key)
            if not np.isfinite(x).all():raise ValueError('Nonfinite '+key)
            result[key]=x.copy()
    if np.any(result['density_sigma']<0):raise ValueError('Negative standard uncertainty')
    return result

def configuration(path):
    c=json.loads(Path(path).read_text(encoding='utf8'))
    for key in KEYS:
        q=c['quantities'][key]
        for name in ['ncc_boundary','nrmse_boundary']:
            v=q[name]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v):raise ValueError('Invalid threshold '+key+'/'+name)
            if name=='ncc_boundary' and not -1<=v<=1:raise ValueError('NCC threshold outside[-1,1]')
            if name=='nrmse_boundary' and v<0:raise ValueError('Negative error threshold')
    v=c['forward_rms_boundary']
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0:raise ValueError('Invalid forward threshold')
    return c

def reference_metrics(x,y):
    # Scale cosine before taking norms to avoid avoidable finite-input overflow.
    sx=np.max(np.abs(x));sy=np.max(np.abs(y))
    if sy==0 or np.ptp(y)==0:raise ValueError('Degenerate evaluator reference')
    cosine=0. if sx==0 else float(np.vdot(x/sx,y/sy).real/(np.linalg.norm(x/sx)*np.linalg.norm(y/sy)))
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        error=float(np.sqrt(np.mean(((x-y)/np.ptp(y))**2)))
    if not np.isfinite(cosine) or not np.isfinite(error):raise ValueError('Nonfinite metric')
    return float(np.clip(cosine,-1,1)),error

def forward_rms(density,raw_path):
    with np.load(raw_path,allow_pickle=False) as f:
        pos=f['positions'];u=f['displacement'];s=f['displacement_sigma']
        if any(a.shape!=(1,32,32,2) or a.dtype!=np.dtype('float64') or not np.isfinite(a).all() for a in [pos,u,s]) or np.any(s<=0):raise ValueError('Invalid evaluator calibration input')
    p=1e-5;M=.105/(1-.105);Z=.25;L=.01;K=.225e-3;n0=1+K*1.225
    scale=n0*p/(M*Z*L*K);grad=u[0]*scale
    variance=(s[0]*scale)**2+(grad*.1/M)**2+(grad*.001/Z)**2
    x=(pos[0,...,0]-256)*p/(M/.75);y=(pos[0,...,1]-256)*p/(M/.75)
    dx=x[1,1]-x[0,0];dy=y[1,1]-y[0,0]
    if dx<=0 or dy<=0:raise ValueError('Invalid evaluator grid')
    v=density[0]
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        rx=(np.diff(v,axis=1)/dx-.5*(grad[:,:-1,0]+grad[:,1:,0]))/np.sqrt(.25*(variance[:,:-1,0]+variance[:,1:,0]))
        ry=(np.diff(v,axis=0)/dy-.5*(grad[:-1,:,1]+grad[1:,:,1]))/np.sqrt(.25*(variance[:-1,:,1]+variance[1:,:,1]))
        mask=np.ones((32,32),bool);mask[1:-1,1:-1]=False
        rb=(v[mask]-1.225)/1e-10
        residual=np.r_[rx.ravel(),ry.ravel(),rb]
        result=float(np.sqrt(np.mean(residual**2)))
    if not np.isfinite(result):raise ValueError('Nonfinite forward residual')
    return result

def evaluate(prediction,evaluation_dir=None):
    root=Path(evaluation_dir) if evaluation_dir is not None else Path(__file__).resolve().parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=configuration(root/'metrics.json')
        # Actual truth always takes precedence; invalid truth never falls back.
        refpath=root/'ground_truth.npz'
        if not refpath.exists():refpath=root/'baseline_reference.npz'
        ref=arrays(refpath);pred=arrays(prediction);metrics={};thresholds={};passed=True
        for key in KEYS:
            cosine,error=reference_metrics(pred[key],ref[key]);metrics[key+'_ncc']=cosine;metrics[key+'_nrmse']=error
            for suffix,comparison,value in [('ncc','>=',cosine),('nrmse','<=',error)]:
                boundary=config['quantities'][key][suffix+'_boundary'];thresholds[key+'_'+suffix]=dict(comparison=comparison,boundary=boundary)
                passed=passed and (value>=boundary if comparison=='>=' else value<=boundary)
        value=forward_rms(pred['reconstruction'],root.parent/'data/raw_data.npz');metrics['forward_rms']=value
        boundary=config['forward_rms_boundary'];thresholds['forward_rms']=dict(comparison='<=',boundary=boundary);passed=passed and value<=boundary
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds)
    except Exception as e:report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(e)])
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Report write failure: '+str(e),file=sys.stderr);return 2
    return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2
if __name__=='__main__':raise SystemExit(main())
