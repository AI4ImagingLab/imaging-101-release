"""Independent standalone signed radar-wavefield evaluator."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def load(path):
    """Validate exact submission shape, dtype, and finite signed amplitudes."""
    if not Path(path).is_file():raise ValueError(f'Missing NPZ: {path}')
    with np.load(path,allow_pickle=False) as d:
        if 'reconstruction' not in d:raise ValueError('Missing reconstruction key')
        a=d['reconstruction']
        if a.shape!=(1,458,1040):raise ValueError(f'Expected (1,458,1040), got {a.shape}')
        if a.dtype!=np.float64:raise ValueError('Expected real float64 wavefield')
        if not np.isfinite(a).all():raise ValueError('Nonfinite wavefield')
        return a

def metrics(estimate,reference):
    """Uncentred cosine NCC and dynamic-range-normalized RMS error."""
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        norm=np.linalg.norm(estimate)*np.linalg.norm(reference);span=np.ptp(reference)
        if span==0 or np.linalg.norm(reference)==0:raise ValueError('Degenerate reference')
        ncc=0. if norm==0 else np.vdot(estimate,reference).real/norm
        result={'ncc':float(np.clip(ncc,-1,1)),'nrmse':float(np.sqrt(np.mean((estimate-reference)**2))/span)}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('Metric computation overflowed; finite prediction amplitudes are not numerically scoreable')
    return result

def evaluate(prediction,assets=HERE):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(assets);config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        ncc=config['ncc_boundary'];nrmse=config['nrmse_boundary']
        if not (np.isfinite(ncc) and -1<=ncc<=1 and np.isfinite(nrmse) and nrmse>0):raise ValueError('Invalid metric boundaries')
        report['thresholds']={'ncc':{'comparison':'>=','boundary':ncc},'nrmse':{'comparison':'<=','boundary':nrmse}}
        ref=assets/'ground_truth.npz'
        if not ref.exists():ref=assets/'baseline_reference.npz'
        reference=load(ref);pred=load(prediction);values=metrics(pred,reference)
        passed=values['ncc']>=ncc and values['nrmse']<=nrmse
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=values)
        return report,0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)];return report,2

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True)
    a=p.parse_args(argv);report,code=evaluate(a.prediction)
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:print(f'Report write error: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
