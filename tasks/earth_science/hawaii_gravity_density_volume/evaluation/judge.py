"""Standalone density-volume and independently observed-gravity evaluator."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ASSETS=Path(__file__).resolve().parent
SHAPE=(1,8,16,16)
MODE='density'
def score(prediction,reference,calibration,observations):
    x=prediction.ravel();r=reference.ravel();span=np.ptp(r);norm=np.linalg.norm(x)*np.linalg.norm(r)
    if span<=0 or np.linalg.norm(r)<=0:raise ValueError('Degenerate hidden density reference')
    latitude=observations['latitude'][0];longitude=observations['longitude'][0];height=observations['elevation'][0]
    s=np.sin(latitude*np.pi/180)**2
    normal=978032.53359*(1+.00193185265241*s)/np.sqrt(1-.00669437999013*s)
    reduced=observations['observed_gravity'][0]-normal+.3086*height-2*np.pi*6.67430e-11*2670.*height*1e5
    east=6371000.*np.cos(19.4*np.pi/180)*(longitude+155.275)*np.pi/180
    north=6371000.*(latitude-19.4)*np.pi/180
    nuisance=np.column_stack([np.ones(len(east)),east/10000.,north/10000.])
    # Independent nuisance projection: QR, rather than the solver's lstsq.
    q,_=np.linalg.qr(nuisance,mode='reduced');observed=-(reduced-q@(q.T@reduced))
    prediction_gravity=calibration['kernel']@x
    prediction_gravity-=q@(q.T@prediction_gravity)
    metrics=dict(density_ncc=float(np.dot(x,r)/norm) if norm>0 else 0.,density_nrmse=float(np.sqrt(np.mean((x-r)**2))/span),observation_rms_mgal=float(np.sqrt(np.mean((prediction_gravity-observed)**2))))
    if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite evaluation metric')
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
        configuration=json.loads((ASSETS/'metrics.json').read_text(encoding='utf-8'));thresholds=read_thresholds(configuration,['density_ncc', 'density_nrmse', 'observation_rms_mgal'])
        if set(thresholds)!={'density_ncc','density_nrmse','observation_rms_mgal'}:raise ValueError('Undefined threshold configuration')
        for key,item in thresholds.items():
            if item['comparison'] not in ('>=','<=') or not np.isfinite(item['boundary']):raise ValueError('Invalid boundary '+key)
        report['thresholds']=thresholds
        with np.load(ASSETS/'baseline_reference.npz',allow_pickle=False) as f:cal={k:f[k].copy() for k in f.files}
        with np.load(ASSETS.parent/'data/raw_data.npz',allow_pickle=False) as f:obs={k:f[k].copy() for k in f.files}
        ref=cal['reconstruction']
        if ref.shape!=SHAPE or ref.dtype!=np.float64 or not np.all(np.isfinite(ref)) or np.ptp(ref)==0:raise ValueError('Invalid hidden density')
        if cal['kernel'].shape!=(1019,2048) or not np.all(np.isfinite(cal['kernel'])):raise ValueError('Invalid acquisition kernel')
    except Exception as exc:
        report['errors']=['Evaluator assets: '+str(exc)];return report,2
    try:
        with np.load(prediction_path,allow_pickle=False) as f:
            if 'reconstruction' not in f:raise ValueError('Missing reconstruction key')
            prediction=f['reconstruction'].copy()
        if prediction.shape!=SHAPE:raise ValueError('Expected shape '+str(SHAPE))
        if prediction.dtype!=np.float64:raise ValueError('Expected float64 real density')
        if not np.all(np.isfinite(prediction)):raise ValueError('Nonfinite prediction')
        metrics=score(prediction,ref,cal,obs)
    except Exception as exc:
        report['errors']=['Prediction: '+str(exc)];return report,2
    report['metrics']=metrics
    for key,item in thresholds.items():
        value=metrics[key];bound=item['boundary']
        if not (value>=bound if item['comparison']=='>=' else value<=bound):report['errors'].append(key+' outside boundary')
    report['passed']=not report['errors'];report['status']='pass' if report['passed'] else 'fail'
    return report,0 if report['passed'] else 1
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    result,code=evaluate(args.prediction)
    try:
        destination=Path(args.report);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:
        print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    print(result['status']);return code
if __name__=='__main__':sys.exit(main())
