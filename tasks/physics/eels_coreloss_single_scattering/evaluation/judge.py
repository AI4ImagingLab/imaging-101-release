"""Independent strict evaluator: never imports the solver or submitted code."""
from pathlib import Path
import argparse,json,sys
import numpy as np
def evaluate(prediction,asset_root=None):
    root=Path(asset_root) if asset_root is not None else Path(__file__).resolve().parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        thresholds=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
        expected={'nrmse_boundary','relative_integral_error_boundary'}
        if not isinstance(thresholds,dict) or {k for k in thresholds if k.endswith('_boundary')}!=expected:
            raise ValueError('Invalid evaluator metric set')
        if thresholds.get('comparisons',{'nrmse':'<=','relative_integral_error':'<='})!={'nrmse':'<=','relative_integral_error':'<='}:
            raise ValueError('Invalid evaluator comparison directions')
        if any(type(thresholds[k]) not in (int,float) for k in expected):
            raise ValueError('Evaluator boundaries must be JSON numbers, excluding booleans')
        with np.load(root/'baseline_reference.npz',allow_pickle=False) as f:ref=f['reconstruction'].copy()
        if ref.shape!=(1,844) or ref.dtype.kind!='f' or not np.isfinite(ref).all():raise ValueError('Invalid hidden reference')
        if np.ptp(ref)<=0 or abs(ref.sum())<=0:raise ValueError('Undefined hidden reference normalization')
        for key in ['nrmse_boundary','relative_integral_error_boundary']:
            if not np.isfinite(thresholds[key]) or thresholds[key]<=0:raise ValueError('Invalid metric boundary')
    except Exception as e:
        report['errors']=['Evaluator assets: '+str(e)];return report,2
    try:
        with np.load(Path(prediction),allow_pickle=False) as f:
            if 'reconstruction' not in f:raise ValueError('Missing reconstruction key')
            pred=f['reconstruction'].copy()
        if pred.shape!=(1,844):raise ValueError('Expected reconstruction shape (1,844)')
        if pred.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Expected real float32 or float64')
        if not np.isfinite(pred).all():raise ValueError('Reconstruction contains NaN or Infinity')
    except Exception as e:
        report['errors']=['Prediction: '+str(e)];return report,2
    pred=pred.astype(np.float64)
    metrics={'nrmse':float(np.sqrt(np.mean((pred-ref)**2))/np.ptp(ref)),'relative_integral_error':float(abs(pred.sum()-ref.sum())/abs(ref.sum()))}
    if not all(np.isfinite(x) for x in metrics.values()):report['errors']=['Nonfinite metric'];return report,2
    report['thresholds']={k:{'comparison':'<=','boundary':thresholds[k+'_boundary']} for k in metrics}
    passed=all(v<=thresholds[k+'_boundary'] for k,v in metrics.items())
    report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics)
    return report,0 if passed else 1
def main():
    if hasattr(sys.stderr,'reconfigure'):sys.stderr.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    report,code=evaluate(a.prediction)
    try:
        target=Path(a.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as e:print('Cannot write report: '+str(e),file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
