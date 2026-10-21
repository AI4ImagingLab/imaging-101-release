"""Standalone deterministic magnetic-source evaluator, independent of solver."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE=Path(__file__).resolve().parent

def validate(path):
    """Read declared physical arrays without changing dtype or shape."""
    if not Path(path).is_file():raise ValueError(f'Missing NPZ: {path}')
    with np.load(path,allow_pickle=False) as d:
        result={}
        for key in ('positions','moments'):
            if key not in d:raise ValueError(f'Missing key: {key}')
            a=d[key]
            if a.shape!=(1,75,3):raise ValueError(f'{key}: expected shape (1,75,3), got {a.shape}')
            if a.dtype!=np.dtype('float64'):raise ValueError(f'{key}: expected real float64')
            if not np.isfinite(a).all():raise ValueError(f'{key}: nonfinite values')
            result[key]=a[0]
    return result

def score(prediction,reference):
    """Optimal geometric assignment; no rotation, translation or rescaling."""
    delta=prediction['positions'][:,None,:]-reference['positions'][None,:,:]
    rows,cols=linear_sum_assignment(np.sum(delta*delta,axis=2))
    p=prediction['positions'][rows];r=reference['positions'][cols]
    m=prediction['moments'][rows];q=reference['moments'][cols]
    norm=np.linalg.norm(q)
    if norm==0:raise ValueError('Degenerate hidden magnetic reference')
    return {'position_rmse_um':float(np.sqrt(np.mean((p-r)**2))),
            'moment_relative_l2':float(np.linalg.norm(m-q)/norm)}

def evaluate(prediction,assets=HERE):
    """Return report, exit code. Hidden truth takes precedence if present."""
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(assets)
        config=json.loads((assets/'metrics.json').read_text(encoding='utf-8'))
        thresholds={k:{'comparison':'<=','boundary':config[k+'_boundary']} for k in ('position_rmse_um','moment_relative_l2')}
        if any(not np.isfinite(v['boundary']) or v['boundary']<=0 for v in thresholds.values()):raise ValueError('Invalid metric boundaries')
        report['thresholds']=thresholds
        ref=assets/'ground_truth.npz'
        if not ref.exists():ref=assets/'baseline_reference.npz'
        reference=validate(ref);pred=validate(prediction)
        metrics=score(pred,reference)
        passed=all(metrics[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)];return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    a=parser.parse_args(argv);report,code=evaluate(a.prediction)
    try:
        p=Path(a.report);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
