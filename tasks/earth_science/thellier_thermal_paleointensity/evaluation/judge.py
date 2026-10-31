"""Independent strict field-vector evaluator. Does not import solver modules."""
from pathlib import Path
import argparse,json,sys,numpy as np
HERE=Path(__file__).resolve().parent

def load(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('Missing reconstruction')
        a=z['reconstruction']
    if a.shape!=(1,5) or a.dtype!=np.dtype('float64'):raise ValueError('Require real float64 shape (1,5)')
    if not np.isfinite(a).all():raise ValueError('Nonfinite field')
    return a

def evaluate(path,assets=HERE):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(assets);config=json.loads((assets/'metrics.json').read_text(encoding='utf8'));boundary=config['field_max_error_boundary']
        if not np.isfinite(boundary) or boundary<=0:raise ValueError('Invalid boundary')
        reference=assets/'ground_truth.npz'
        if not reference.exists():reference=assets/'baseline_reference.npz'
        ref=load(reference);a=load(path);error=float(np.max(np.abs(a-ref)));passed=error<=boundary
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=dict(field_max_error=error),thresholds=dict(field_max_error=dict(comparison='<=',boundary=boundary)))
    except Exception as exc:report['errors']=[str(exc)]
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args();report=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print(str(exc),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[report['status']]

if __name__=='__main__':sys.exit(main())
