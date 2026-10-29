"""Independent height-vector evaluator; no solver imports."""
from pathlib import Path
import argparse,json,sys,numpy as np
HERE=Path(__file__).resolve().parent
def load(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('Missing reconstruction')
        a=z['reconstruction']
    if a.shape!=(1,324) or a.dtype!=np.dtype('float64'):raise ValueError('Require float64 shape(1,324)')
    if not np.isfinite(a).all():raise ValueError('Nonfinite height')
    return a
def evaluate(path,assets=HERE):
    r=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(assets);c=json.loads((assets/'metrics.json').read_text());b=c['height_max_error_boundary']
        if not np.isfinite(b) or b<=0:raise ValueError('Invalid boundary')
        ref=assets/'ground_truth.npz'
        if not ref.exists():ref=assets/'baseline_reference.npz'
        truth=load(ref);a=load(path);error=float(np.max(abs(a-truth)))
        passed=bool(error<=b and np.all((a>.5)&(a<=8)))
        r.update(status='pass' if passed else 'fail',passed=passed,metrics={'height_max_error':error},thresholds={'height_max_error':{'comparison':'<=','boundary':b}})
    except Exception as e:r['errors']=[str(e)]
    return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();r=evaluate(a.prediction)
    try:
        q=Path(a.report);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print(str(e),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[r['status']]
if __name__=='__main__':sys.exit(main())
