"""Standalone evaluation, independent of cleaned solver modules."""
import argparse,json,sys
from pathlib import Path
import numpy as np

def load_prediction(path):
    if not Path(path).is_file():raise ValueError(f'Missing prediction/reference file: {path}')
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('Missing reconstruction key')
        a=z['reconstruction']
        if a.shape!=(1,101,51):raise ValueError('Expected shape (1,101,51)')
        if a.dtype!=np.dtype('float64'):raise ValueError('Expected real float64')
        if not np.isfinite(a).all():raise ValueError('Nonfinite reconstruction')
        return a.copy()

def observation_rms(field,data_path):
    """Independent direct-row kernel evaluation with bounded memory."""
    with np.load(data_path,allow_pickle=False) as z:
        f1=z['radial_kernel_1'][0];f2=z['radial_kernel_2'][0]
        g1=z['angular_kernel_1'][0];g2=z['angular_kernel_2'][0]
        ii=z['radial_index'][0];jj=z['angular_index'][0]
        observed=z['splittings'][0];sigma=np.abs(z['splitting_error'][0])
    prediction=np.empty(len(ii));x=field[0]
    for lo in range(0,len(ii),128):
        end=min(lo+128,len(ii));a=ii[lo:end];b=jj[lo:end]
        kernel=f1[:,a].T[:,:,None]*g1[:,b].T[:,None,:]+f2[:,a].T[:,:,None]*g2[:,b].T[:,None,:]
        prediction[lo:end]=np.einsum('brt,rt->b',kernel,x)
    sigma=np.where(sigma==0,1.,sigma)
    return float(np.sqrt(np.mean(((prediction-observed)/sigma)**2)))

def evaluate(prediction,asset_dir=None,data_path=None):
    root=Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        settings=json.loads((root/'metrics.json').read_text(encoding='utf8'))
        ref=load_prediction(root/'baseline_reference.npz');p=load_prediction(prediction)
        raw=Path(data_path) if data_path is not None else root.parent/'data/raw_data.npz'
        thresholds={k:dict(comparison='<=',boundary=settings[k+'_boundary']) for k in ['rotation_max_abs_nHz','observation_weighted_rms']}
        if any(not np.isfinite(v['boundary']) or v['boundary']<0 for v in thresholds.values()):raise ValueError('Invalid metric boundaries')
        measured=dict(rotation_max_abs_nHz=float(np.max(abs(p-ref))),observation_weighted_rms=observation_rms(p,raw))
        passed=all(measured[k]<=thresholds[k]['boundary'] for k in measured)
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=measured,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)];return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
