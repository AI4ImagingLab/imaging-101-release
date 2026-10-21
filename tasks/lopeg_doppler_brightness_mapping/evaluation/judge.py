"""Independent NPZ contract and spatial/measurement evaluator; no solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
MODE='brightness'
ASSETS=Path(__file__).resolve().parent
SHAPE=(1,4584) if MODE=='brightness' else (1,3,4584)

def score(prediction,reference,calibration,observations):
    metrics={}
    arrays=[('brightness',prediction[0],reference[0])] if MODE=='brightness' else [(n,p,r) for n,p,r in zip(['radial','colatitudinal','azimuthal'],prediction[0],reference[0])]
    for name,p,r in arrays:
        span=np.ptp(r);norm=np.linalg.norm(p)*np.linalg.norm(r)
        if span<=0 or np.linalg.norm(r)<=0:raise ValueError('Degenerate reference '+name)
        metrics[name+'_ncc']=float(np.vdot(p,r).real/norm) if norm>0 else 0.
        metrics[name+'_nrmse']=float(np.sqrt(np.mean((p-r)**2))/span)
    weights=calibration['weights'];views=calibration['view']
    if MODE=='brightness':
        intensity=prediction[0]
        weighted=weights*intensity[None,:]
        profiles=np.einsum('ovc,oc->ov',calibration['local_i'],weighted)/weighted.sum(axis=1)[:,None]
        observed=observations['stokes_i'][0];sigma=observations['sigma_i'][0]
    else:
        br,bt,bp=prediction[0];theta=calibration['colatitude'];phi=calibration['longitude']
        st,ct,sp,cp=np.sin(theta),np.cos(theta),np.sin(phi),np.cos(phi)
        cart=np.array([br*st*cp+bt*ct*cp-bp*sp,br*st*sp+bt*ct*sp+bp*cp,br*ct-bt*st])
        line_of_sight=np.einsum('oic,ic->oc',views,cart)
        profiles=np.einsum('ovc,oc->ov',calibration['local_v'],weights*line_of_sight)/weights.sum(axis=1)[:,None]
        observed=observations['stokes_v'][0];sigma=observations['sigma_v'][0]
    metrics['observation_chi2_per_sample']=float(np.mean(((profiles-observed)/sigma)**2))
    if not all(np.isfinite(x) for x in metrics.values()):raise ValueError('Nonfinite metric')
    return metrics

def read_thresholds(configuration,required_metrics):
    """Read each canonical named boundary exactly once; directions are separate."""
    comparisons=configuration["metric_comparisons"]
    if set(comparisons)!=set(required_metrics):raise ValueError("Undefined metric comparison configuration")
    named={key[:-9] for key in configuration if key.endswith("_boundary")}
    if named!=set(required_metrics):raise ValueError("Missing or unexpected named metric boundary")
    thresholds={}
    for metric in required_metrics:
        comparison=comparisons[metric];boundary=configuration[metric+"_boundary"]
        expected=">=" if metric.endswith("_ncc") else "<="
        if comparison!=expected or not np.isfinite(boundary):raise ValueError("Invalid named boundary "+metric)
        thresholds[metric]={"comparison":comparison,"boundary":boundary}
    return thresholds

def evaluate(prediction_path):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((ASSETS/'metrics.json').read_text(encoding='utf-8'))
        thresholds=read_thresholds(config,['brightness_ncc', 'brightness_nrmse', 'observation_chi2_per_sample'])
        for name,item in thresholds.items():
            if item['comparison'] not in ('>=','<=') or not np.isfinite(item['boundary']):raise ValueError('Invalid threshold '+name)
        report['thresholds']=thresholds
        with np.load(ASSETS/'baseline_reference.npz',allow_pickle=False) as f:cal={k:f[k].copy() for k in f.files}
        with np.load(ASSETS.parent/'data/raw_data.npz',allow_pickle=False) as f:observed={k:f[k].copy() for k in f.files}
        reference=cal['reconstruction']
        if reference.shape!=SHAPE or reference.dtype!=np.float64 or not np.all(np.isfinite(reference)):raise ValueError('Invalid hidden reference')
    except Exception as exc:
        report['errors']=['Evaluator assets: '+str(exc)];return report,2
    try:
        with np.load(prediction_path,allow_pickle=False) as f:
            if 'reconstruction' not in f:raise ValueError('Missing reconstruction key')
            prediction=f['reconstruction'].copy()
        if prediction.shape!=SHAPE:raise ValueError('Shape must be '+str(SHAPE))
        if prediction.dtype!=np.float64:raise ValueError('Dtype must be float64')
        if not np.all(np.isfinite(prediction)):raise ValueError('Nonfinite prediction')
        if MODE=='brightness' and np.any(prediction<=0):raise ValueError('Brightness must be positive')
        metrics=score(prediction,reference,cal,observed)
    except Exception as exc:
        report['errors']=['Prediction: '+str(exc)];return report,2
    report['metrics']=metrics
    for name,item in thresholds.items():
        value=metrics[name];boundary=item['boundary']
        if not (value>=boundary if item['comparison']=='>=' else value<=boundary):report['errors'].append(name+' outside boundary')
    report['passed']=not report['errors'];report['status']='pass' if report['passed'] else 'fail'
    return report,0 if report['passed'] else 1

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    report,code=evaluate(a.prediction)
    try:
        destination=Path(a.report);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:
        print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    print(report['status']);return code
if __name__=='__main__':sys.exit(main())
