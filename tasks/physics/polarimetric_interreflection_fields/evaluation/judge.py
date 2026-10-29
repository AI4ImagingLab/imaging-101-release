"""Independent standalone evaluator; explicit trigonometric observation equation."""
from pathlib import Path
import argparse,json,sys
import numpy as np
BASE=Path(__file__).resolve().parent
CHANNELS=('unpolarized','sum_real','sum_imaginary','difference_real','difference_imaginary')
SHAPE=(1,5,2048,2448)

def load_prediction(path):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'reconstruction'}:raise ValueError('Exactly reconstruction required')
        a=z['reconstruction']
    if a.shape!=SHAPE:raise ValueError('Expected shape'+str(SHAPE))
    if a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('float32 or float64 real reconstruction required')
    if not np.isfinite(a).all():raise ValueError('Nonfinite reconstruction')
    return a[0]

def observation_error(fields):
    root=BASE.parent
    with np.load(root/'data/raw_data.npz',allow_pickle=False) as z:
        y=z['intensity'];g=z['polarizer_psg'];c=z['polarizer_psa']
    if y.shape!=(1,16,2048,2448) or g.shape!=(1,16) or c.shape!=(1,16):raise ValueError('Evaluator calibration shape invalid')
    if not np.isfinite(y).all() or not np.isfinite(g).all() or not np.isfinite(c).all():raise ValueError('Invalid evaluator calibration')
    g=g[0,:,None,None];c=c[0,:,None,None];error=0.;norm=0.
    # Independent angular harmonic formula, without importing runtime design/field helpers.
    for k in range(0,2048,64):
        u,sr,si,dr,di=fields[:,k:k+64].astype(np.float64)
        mean=.5*(u+np.hypot(sr,si)+np.hypot(dr,di))
        predicted=mean[None]+.5*(sr[None]*np.cos(2*(c+g))+si[None]*np.sin(2*(c+g))+dr[None]*np.cos(2*(c-g))+di[None]*np.sin(2*(c-g)))
        observed=np.clip(y[0,:,k:k+64],0,None).astype(np.float64)
        error+=float(np.sum((predicted-observed)**2));norm+=float(np.sum(observed**2))
    if norm<=0:raise ValueError('Degenerate observation norm')
    return float(np.sqrt(error/norm))

def evaluate(path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        data=json.loads((BASE/'metrics.json').read_text(encoding='utf-8'))
        bounds={channel+'_relative_l2':data[channel+'_relative_l2_boundary'] for channel in CHANNELS}
        bounds['observation_relative_l2']=data['observation_relative_l2_boundary']
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for v in bounds.values()):raise ValueError('Invalid evaluator thresholds')
        report['thresholds']={k:{'comparison':'<=','boundary':v} for k,v in bounds.items()}
        truth=BASE/'ground_truth.npz';reference=load_prediction(truth if truth.exists() else BASE/'baseline_reference.npz')
        norms=[float(np.linalg.norm(a)) for a in reference]
        if any(n<=0 for n in norms):raise ValueError('Degenerate evaluator channel')
        prediction=load_prediction(path)
        values={channel+'_relative_l2':float(np.linalg.norm(a.astype(np.float64)-r)/norm) for channel,a,r,norm in zip(CHANNELS,prediction,reference,norms)}
        values['observation_relative_l2']=observation_error(prediction)
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Undefined metric')
        report.update(metrics=values,passed=all(values[k]<=bounds[k] for k in bounds));report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as exc:
        report.update(metrics={},status='error',passed=False,errors=[str(exc)]);return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report,code=evaluate(Path(args.prediction))
    try:
        destination=Path(args.report);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
