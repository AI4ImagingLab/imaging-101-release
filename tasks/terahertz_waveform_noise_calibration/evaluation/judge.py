"""Standalone evaluator: explicit declared arrays, no solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
def relative_error(estimate,reference):
    norm=float(np.linalg.norm(reference.ravel()))
    if not np.isfinite(norm) or norm<=0:raise ValueError('Evaluator reference has zero or invalid norm')
    with np.errstate(over='ignore',invalid='ignore'):
        value=float(np.linalg.norm((estimate-reference).ravel())/norm)
    if not np.isfinite(value):raise ValueError('Prediction relative error overflowed or is nonfinite')
    return value
def read_arrays(path,schema,label):
    if not Path(path).is_file():raise ValueError(f'{label} file does not exist: {path}')
    with np.load(path,allow_pickle=False) as f:
        result={}
        for key,spec in schema.items():
            if key not in f:raise ValueError(f'{label} missing required key {key}')
            v=f[key]
            if list(v.shape)!=spec['shape']:raise ValueError(f'{label} {key} shape {v.shape}, expected {spec["shape"]}')
            if str(v.dtype)!=spec['dtype']:raise ValueError(f'{label} {key} dtype {v.dtype}, expected {spec["dtype"]}')
            if not np.isfinite(v).all():raise ValueError(f'{label} {key} contains NaN or infinity')
            result[key]=v
    return result
def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'))
        schema=config['output_schema']
        groups=config['metric_groups']
        if not schema or not isinstance(groups,list) or not groups:raise ValueError('Evaluator has no declared output schema or metric groups')
        if {group.get('key') for group in groups}!=set(schema):raise ValueError('Evaluator metrics do not cover every declared output')
        for group in groups:
            boundary=group.get('boundary')
            if not isinstance(boundary,(int,float)) or not np.isfinite(boundary) or boundary<0:raise ValueError('Evaluator boundary must be a finite nonnegative number')
        p=read_arrays(prediction,schema,'Prediction')
        target=ROOT/'ground_truth.npz'
        if not target.exists():target=ROOT/'baseline_reference.npz'
        r=read_arrays(target,schema,'Evaluator reference')
        metrics={};thresholds={}
        for metric in config['metric_groups']:
            key=metric['key'];a=p[key];b=r[key]
            if 'component' in metric:
                axis=metric.get('axis',-1);a=np.take(a,metric['component'],axis=axis);b=np.take(b,metric['component'],axis=axis)
            name=metric['name'];metrics[name]=relative_error(a,b);thresholds[name]={'comparison':'<=','boundary':metric['boundary']}
        passed=all(metrics[k]<=thresholds[k]['boundary'] for k in metrics)
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=thresholds)
    except Exception as error:report['errors']=[str(error)]
    return report
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as error:
        print(f'Cannot write report: {error}',file=sys.stderr);return 2
    return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2
if __name__=='__main__':sys.exit(main())
