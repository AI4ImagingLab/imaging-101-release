"""Independent standalone all-camera NPZ evaluator; imports no solver modules."""
from pathlib import Path
import argparse,json,sys
import numpy as np

CAMERAS=('eos','sony','iphone15promax','dji_pocket3')
SHAPE=(1,4,31,3)

def validate_array(path):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'reconstruction'}:raise ValueError('Expected exactly reconstruction key')
        a=z['reconstruction']
    if a.shape!=SHAPE:raise ValueError(f'Expected shape {SHAPE}; got {a.shape}')
    if a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Expected real float32 or float64')
    if not np.isfinite(a).all():raise ValueError('Array contains NaN or Infinity')
    return a.astype(np.float64,copy=False)

def pair_metrics(estimate,reference):
    dynamic=float(np.ptp(reference))
    if dynamic<=0 or np.linalg.norm(reference)==0:raise ValueError('Degenerate evaluator reference')
    norm=float(np.linalg.norm(estimate)*np.linalg.norm(reference))
    ncc=float(np.vdot(estimate,reference)/norm) if norm else 0.
    return ncc,float(np.sqrt(np.mean((estimate-reference)**2))/dynamic)

def evaluate(prediction,assets=None):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(assets) if assets else Path(__file__).resolve().parent
        reference=validate_array(assets/'ground_truth.npz')
        config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        boundaries={}
        for c in CAMERAS:
            for metric,comparison in [('ncc','>='),('nrmse','<=')]:
                key=c+'_'+metric;value=config[key+'_boundary']
                if not isinstance(value,(int,float)) or isinstance(value,bool) or not np.isfinite(value):raise ValueError('Invalid metric boundary '+key)
                boundaries[key]=dict(comparison=comparison,boundary=float(value))
        gauge=config['peak_error_boundary']
        if not isinstance(gauge,(int,float)) or not np.isfinite(gauge) or gauge<0:raise ValueError('Invalid peak boundary')
        boundaries['peak_error']=dict(comparison='<=',boundary=float(gauge))
        prediction=validate_array(prediction);metrics={}
        for i,c in enumerate(CAMERAS):
            metrics[c+'_ncc'],metrics[c+'_nrmse']=pair_metrics(prediction[0,i],reference[0,i])
        metrics['peak_error']=float(np.max(abs(prediction.max(axis=(2,3))-1)))
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite evaluated metric')
        passed=all(metrics[k]>=v['boundary'] if v['comparison']=='>=' else metrics[k]<=v['boundary'] for k,v in boundaries.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=boundaries)
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(exc)])
    return report

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args(argv);report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print('Report-write failure: '+str(exc),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[report['status']]

if __name__=='__main__':raise SystemExit(main())
