"""Independent full-matrix truth and observation evaluator."""
import argparse,json,sys
from pathlib import Path
import numpy as np

def load_matrix(path):
    if not Path(path).is_file():raise ValueError(f'Missing NPZ file: {path}')
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('Missing reconstruction key')
        x=z['reconstruction']
        if x.shape!=(1,4,4):raise ValueError('Expected shape (1,4,4)')
        if x.dtype!=np.dtype('float64'):raise ValueError('Expected real float64 Mueller matrix')
        if not np.isfinite(x).all():raise ValueError('Nonfinite matrix values')
        return x.copy()

def predict_intensity(matrix,angles):
    """Independent calibrated analyzer/generator Stokes vectors, no solver import."""
    angle=np.asarray(angles);c=np.cos(2*angle);s=np.sin(2*angle)
    ca=np.cos(10*angle);sa=np.sin(10*angle);C=np.cos(np.pi/2);S=np.sin(np.pi/2)
    generator=np.column_stack((np.ones_like(c),c*c+C*s*s,(1-C)*c*s,S*s))/2
    analyzer=np.column_stack((np.ones_like(ca),ca*ca+C*sa*sa,(1-C)*ca*sa,-S*sa))/2
    return np.einsum('ni,ij,nj->n',analyzer,matrix[0],generator)

def score_arrays(prediction,reference,angles,power):
    p=prediction.ravel();r=reference.ravel();scale=np.ptp(r);refnorm=np.linalg.norm(r)
    if scale<=0 or refnorm<=0 or np.linalg.norm(power)<=0:raise ValueError('Degenerate reference metric scale')
    ncc=float(np.dot(p,r)/(np.linalg.norm(p)*refnorm)) if np.linalg.norm(p)>0 else 0.
    return dict(ncc_vs_truth=ncc,nrmse_vs_truth=float(np.sqrt(np.mean((p-r)**2))/scale),observation_relative_l2=float(np.linalg.norm(predict_intensity(prediction,angles)-power)/np.linalg.norm(power)))

def evaluate(prediction,asset_dir=None,data_path=None):
    assets=Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        ref=load_matrix(assets/'ground_truth.npz');x=load_matrix(prediction)
        with np.load(Path(data_path) if data_path is not None else assets.parent/'data/raw_data.npz',allow_pickle=False) as z:
            angles=z['angles'][0];power=z['power'][0]
        if angles.shape!=(24,) or power.shape!=(24,) or not np.isfinite(angles).all() or not np.isfinite(power).all():raise ValueError('Invalid evaluator acquisition')
        thresholds=dict(ncc_vs_truth=dict(comparison='>=',boundary=config['ncc_boundary']),nrmse_vs_truth=dict(comparison='<=',boundary=config['nrmse_boundary']),observation_relative_l2=dict(comparison='<=',boundary=config['observation_relative_l2_boundary']))
        if not all(np.isfinite(v['boundary']) for v in thresholds.values()) or not -1<=config['ncc_boundary']<=1 or config['nrmse_boundary']<0 or config['observation_relative_l2_boundary']<0:raise ValueError('Invalid metric boundaries')
        measured=score_arrays(x,ref,angles,power)
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('Metric numerical overflow')
        passed=measured['ncc_vs_truth']>=config['ncc_boundary'] and measured['nrmse_vs_truth']<=config['nrmse_boundary'] and measured['observation_relative_l2']<=config['observation_relative_l2_boundary']
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=measured,thresholds=thresholds);return report,0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)];return report,2

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args();report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
