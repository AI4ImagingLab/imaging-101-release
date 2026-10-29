"""Independent full-array numerical evaluation; never imports a solver."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPE=(1,1585,886,3)

def load(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('Missing reconstruction')
        a=z['reconstruction']
    if a.shape!=SHAPE:raise ValueError('Wrong reconstruction shape')
    if a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Require real float32 or float64')
    if not np.isfinite(a).all():raise ValueError('Nonfinite reconstruction')
    return a.astype(float,copy=False)

def metrics(x,r):
    span=float(np.ptp(r));rn=np.linalg.norm(r);xn=np.linalg.norm(x)
    if span<=0 or rn<=0:raise ValueError('Degenerate reference')
    return dict(ncc=float(np.sum(x*r)/(xn*rn)) if xn else 0.,nrmse=float(np.sqrt(np.mean((x-r)**2))/span))

def evaluate(prediction,assets=HERE):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(assets);config=json.loads((assets/'metrics.json').read_text())
        ref=assets/'ground_truth.npz'
        if not ref.exists():ref=assets/'baseline_reference.npz'
        r=load(ref);x=load(prediction);measured=metrics(x,r)
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as z:green=z['measurements'][...,1]
        measured['green_max_error']=float(np.max(abs(x[...,1]-green)))
        comparisons={'ncc':'>=','nrmse':'<=','green_max_error':'<='}
        thresholds={k:dict(comparison=op,boundary=config[k+'_boundary']) for k,op in comparisons.items()}
        if not all(np.isfinite(t['boundary']) for t in thresholds.values()):raise ValueError('Nonfinite threshold')
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('Nonfinite metric')
        passed=all(measured[k]>=t['boundary'] if t['comparison']=='>=' else measured[k]<=t['boundary'] for k,t in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured,thresholds=thresholds)
    except Exception as e:report.update(errors=[str(e)],metrics={})
    return report

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);a=parser.parse_args(argv)
    report=evaluate(a.prediction)
    try:
        p=Path(a.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[report['status']]
if __name__=='__main__':sys.exit(main())
