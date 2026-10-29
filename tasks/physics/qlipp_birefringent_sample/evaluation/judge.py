"""Independent standalone NPZ evaluator, with per-quantity contracts."""
from pathlib import Path
import argparse,json,sys
import numpy as np

def score(estimate,reference):
    """Relative L2; reject zero-norm references."""
    norm=np.linalg.norm(reference.astype(np.float64).ravel())
    if norm<=0:raise ValueError('Evaluator reference has zero norm')
    return float(np.linalg.norm((estimate.astype(np.float64)-reference).ravel())/norm)

def angular_score(estimate,reference,retardance):
    """Pi-periodic orientation RMSE, continuously weighted by sin(delta)**2."""
    difference=estimate.astype(np.float64)-reference
    wrapped=np.arctan2(np.sin(2*difference),np.cos(2*difference))/2
    w=np.sin(retardance.astype(np.float64))**2
    if np.sum(w)<=0:raise ValueError('No identifiable orientation reference')
    return float(np.sqrt(np.sum(w*wrapped**2)/np.sum(w)))

def evaluate(prediction,asset_dir=None):
    assets=Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        spec=config['outputs']
        if not spec:raise ValueError('Empty evaluator output configuration')
        for key,item in spec.items():
            b=item['boundary']
            if not np.isfinite(b) or b<=0 or item['metric'] not in ['relative_l2','angular_rmse']:
                raise ValueError('Invalid evaluator metric/boundary')
            report['thresholds'][key+'_'+item['metric']]={'comparison':'<=','boundary':b}
        truth=assets/'ground_truth.npz';refpath=truth if truth.exists() else assets/'baseline_reference.npz'
        with np.load(refpath,allow_pickle=False) as d:ref={k:d[k] for k in spec}
        with np.load(prediction,allow_pickle=False) as d:
            if any(k not in d for k in spec):raise ValueError('Missing required output key')
            estimate={k:d[k] for k in spec}
        for label,data in [('Evaluator reference',ref),('Prediction',estimate)]:
            for key,item in spec.items():
                a=data[key]
                if a.shape!=tuple(item['shape']):raise ValueError(f'{label} {key}: expected shape {item["shape"]}')
                if a.dtype!=np.dtype(item['dtype']) or a.dtype.kind!='f':raise ValueError(f'{label} {key}: expected real {item["dtype"]}')
                if not np.isfinite(a).all():raise ValueError(f'{label} {key}: NaN or Infinity')
                if key=='orientation' and (np.min(a)<0 or np.max(a)>=np.pi):raise ValueError(f'{label}: orientation must be [0,pi)')
        values={}
        for key,item in spec.items():
            value=angular_score(estimate[key],ref[key],ref['reconstruction']) if item['metric']=='angular_rmse' else score(estimate[key],ref[key])
            if not np.isfinite(value):raise ValueError('Undefined evaluator metric')
            values[key+'_'+item['metric']]=value
        report['metrics']=values
        report['passed']=all(v<=report['thresholds'][k]['boundary'] for k,v in values.items())
        report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as e:
        report['metrics']={};report['errors'].append(str(e));return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();report,code=evaluate(args.prediction)
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:
        print(f'Cannot write report: {e}',file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
