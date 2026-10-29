"""Standalone chronology and observed-activity evaluator; no solver imports."""
from pathlib import Path
import argparse,json,sys,numpy as np

KEYS=('reconstruction','sedimentation_rate','initial_activity','modeled_activity')
HERE=Path(__file__).resolve().parent

def validate_arrays(values):
    for key in KEYS:
        if key not in values:raise ValueError('Missing required key: '+key)
        a=values[key]
        if a.shape!=(1,14):raise ValueError(key+' must have shape (1,14)')
        if a.dtype!=np.float64:raise ValueError(key+' must have dtype float64')
        if not np.all(np.isfinite(a)):raise ValueError(key+' contains NaN/Inf')

def score(prediction,truth,mass,activity,error):
    age,rate,a0,modeled=(prediction[k][0] for k in KEYS)
    ref=truth['reconstruction'][0]
    if ref.shape!=(14,) or not np.all(np.isfinite(ref)) or np.ptp(ref)<=0 or np.linalg.norm(ref)<=0:raise ValueError('Invalid hidden age measurement')
    if np.any(rate<=0) or np.any(a0<=0) or np.any(np.diff(np.r_[0.,age])<=0):
        raise ValueError('Age increments, sedimentation rates and initial activities must be positive')
    if mass.shape!=(14,) or activity.shape!=(14,) or error.shape!=(14,) or np.any(error<=0) or not np.all(np.isfinite([mass,activity,error])):raise ValueError('Invalid public observation calibration')
    dm=np.diff(np.r_[0.,mass]);duration=np.diff(np.r_[0.,age]);previous=np.r_[0.,age[:-1]]
    x=.03118*duration
    predicted=a0*np.exp(-.03118*previous)*(1-np.exp(-x))/x
    norm=np.linalg.norm(age)
    if norm<=0:raise ValueError('Zero chronology norm')
    metrics={'age_ncc':float(np.dot(age,ref)/(norm*np.linalg.norm(ref))),
             'age_nrmse':float(np.sqrt(np.mean((age-ref)**2))/np.ptp(ref)),
             'observation_chi':float(np.sqrt(np.sum(((modeled-activity)/error)**2)/10)),
             'age_rate_relative_consistency':float(np.max(abs(rate*duration-dm)/dm)),
             'decay_relative_consistency':float(np.max(abs(modeled-predicted)/np.maximum(abs(predicted),1.)))}
    if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite metric')
    return metrics

def evaluate(prediction_path,asset_dir=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        asset_dir=Path(asset_dir)
        configuration=json.loads((asset_dir/'metrics.json').read_text(encoding='utf8'))
        rules=configuration['thresholds'];report['thresholds']=rules
        if not rules or any(r['comparison'] not in ['>=','<='] or not np.isfinite(r['boundary']) for r in rules.values()):raise ValueError('Invalid thresholds')
        with np.load(prediction_path,allow_pickle=False) as f:prediction={k:f[k] for k in f.files}
        validate_arrays(prediction)
        with np.load(asset_dir/'ground_truth.npz',allow_pickle=False) as f:truth={k:f[k] for k in f.files}
        with np.load(asset_dir.parent/'data/raw_data.npz',allow_pickle=False) as f:mass,activity,error=(f[k][0] for k in ['mass_depth','activity','activity_error'])
        metrics=score(prediction,truth,mass,activity,error)
        if set(metrics)!=set(rules):raise ValueError('Metric/threshold mismatch')
        passed=all(metrics[k]>=r['boundary'] if r['comparison']=='>=' else metrics[k]<=r['boundary'] for k,r in rules.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics)
    except Exception as exc:report.update(status='error',passed=False,metrics={},errors=[str(exc)])
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    result=evaluate(args.prediction)
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report-write failure: '+str(exc),file=sys.stderr);return 2
    return 0 if result['passed'] else 1 if result['status']=='fail' else 2

if __name__=='__main__':sys.exit(main())
