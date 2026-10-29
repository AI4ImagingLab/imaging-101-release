"""Standalone nine-condition screen-geometry combination and measured-power Judge."""
from pathlib import Path
import argparse,json,sys,numpy as np
HERE=Path(__file__).resolve().parent

def score(prediction,reference):
    values=prediction['reconstruction']
    if values.shape!=(1,9):raise ValueError('reconstruction must have shape (1,9)')
    if values.dtype!=np.float64:raise ValueError('reconstruction must have dtype float64')
    if not np.all(np.isfinite(values)) or np.any(values<=0):raise ValueError('All nine physical kappas must be finite and positive')
    target=reference['reconstruction']
    if target.shape!=(1,9) or target.dtype!=np.float64 or not np.all(np.isfinite(target)) or np.any(target<=0):raise ValueError('Invalid hidden standard-method reference')
    relative=float(np.max(abs(values-target)/target));deficits=[]
    for i,kappa in enumerate(values[0]):
        grid=reference[f'curvature_grid_{i}'];power=reference[f'observed_power_{i}']
        if grid.ndim!=1 or grid.shape!=power.shape or len(grid)<3 or not np.all(np.isfinite([grid,power])) or np.any(np.diff(grid)<=0) or np.ptp(power)<=0:raise ValueError('Invalid hidden observed arc-power profile')
        predicted_power=np.interp(kappa/2e6,grid,power) if grid[0]<=kappa/2e6<=grid[-1] else np.max(power)-1e100
        deficits.append(float(np.max(power)-predicted_power))
    result={'maximum_kappa_relative_error':relative,'maximum_power_deficit_db':max(deficits)}
    if not all(np.isfinite(v) for v in result.values()):raise ValueError('Nonfinite metric')
    return result

def evaluate(prediction_path,asset_dir=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        asset_dir=Path(asset_dir);configuration=json.loads((asset_dir/'metrics.json').read_text(encoding='utf8'));rules=configuration['thresholds'];report['thresholds']=rules
        if not rules or any(r['comparison']!='<=' or not np.isfinite(r['boundary']) or r['boundary']<0 for r in rules.values()):raise ValueError('Invalid evaluator thresholds')
        with np.load(prediction_path,allow_pickle=False) as f:prediction={k:f[k] for k in f.files}
        if 'reconstruction' not in prediction:raise ValueError('Missing required key: reconstruction')
        with np.load(asset_dir/'baseline_reference.npz',allow_pickle=False) as f:reference={k:f[k] for k in f.files}
        # Observed arc profiles are fixed acquisition-derived evaluator assets;
        # actual physical parameter truth, if independently available, takes
        # precedence over the method parameter result without replacing profiles.
        truth=asset_dir/'ground_truth.npz'
        if truth.exists():
            with np.load(truth,allow_pickle=False) as f:reference['reconstruction']=f['reconstruction'].copy()
            rules=dict(rules);rules['maximum_kappa_relative_error']={'comparison':'<=','boundary':configuration['actual_truth_relative_error_boundary']};report['thresholds']=rules
        metrics=score(prediction,reference)
        if not set(rules).issubset(metrics):raise ValueError('Metric/threshold mismatch')
        metrics={name:metrics[name] for name in rules}
        passed=all(metrics[k]<=r['boundary'] for k,r in rules.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics)
    except Exception as exc:report.update(status='error',passed=False,metrics={},errors=[str(exc)])
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args();report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report-write failure: '+str(exc),file=sys.stderr);return 2
    return 0 if report['passed'] else 1 if report['status']=='fail' else 2

if __name__=='__main__':sys.exit(main())
