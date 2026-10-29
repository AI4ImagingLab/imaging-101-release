"""Independent point-set/profile evaluator; never imports the solver."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.optimize import linear_sum_assignment
HERE=Path(__file__).resolve().parent
PIXEL_NM=.012443924564929288
COUNTS={'a_columns':1009,'b_columns':958,'oxygen_columns':922}
SUFFIXES=('centroid_max_pixel','centroid_rms_pixel','covariance_relative','peak_relative')
def read_arrays(path):
    with np.load(path,allow_pickle=False) as z:
        out={}
        for k,n in COUNTS.items():
            if k not in z:raise ValueError(f'Missing required key: {k}')
            a=z[k]
            if a.shape!=(1,n,6):raise ValueError(f'Invalid shape for {k}: {a.shape}')
            if a.dtype!=np.dtype('float64'):raise ValueError(f'Expected real float64 for {k}')
            if not np.isfinite(a).all():raise ValueError(f'Nonfinite values in {k}')
            out[k]=a[0]
    return out
def invalid_profiles(a):
    with np.errstate(over='raise',invalid='raise'):
        bad=(a[:,2]<=0)|(a[:,4]<=0)|(a[:,2]*a[:,4]-a[:,3]**2<=0)|(a[:,5]<=0)
    return int(bad.sum())
def compare(a,b):
    # Column identity is unordered within each chemical sublattice. Matching
    # changes ordering only: there is no translation, scale or phase fit.
    row,col=linear_sum_assignment(np.linalg.norm(a[:,None,:2]-b[None,:,:2],axis=-1))
    x,y=a[row],b[col]
    d=np.linalg.norm(x[:,:2]-y[:,:2],axis=1)/PIXEL_NM
    nc,npk=np.linalg.norm(y[:,2:5]),np.linalg.norm(y[:,5])
    if nc<=0 or npk<=0:raise ValueError('Degenerate evaluator reference')
    return dict(centroid_max_pixel=float(d.max()),centroid_rms_pixel=float(np.sqrt(np.mean(d*d))),covariance_relative=float(np.linalg.norm(x[:,2:5]-y[:,2:5])/nc),peak_relative=float(np.linalg.norm(x[:,5]-y[:,5])/npk),invalid_profile_count=invalid_profiles(a))
def evaluate(prediction,reference_dir=HERE):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        base=Path(reference_dir)
        config=json.loads((base/'metrics.json').read_text(encoding='utf8'))
        bounds={}
        for name in SUFFIXES:
            v=config[name+'_boundary']
            if type(v) not in (float,int) or not np.isfinite(v) or v<0:raise ValueError(f'Invalid {name} boundary')
            bounds[name]=float(v)
        if type(config['invalid_profile_count_boundary']) not in (float,int) or config['invalid_profile_count_boundary']!=0:raise ValueError('Invalid physical-profile boundary')
        bounds['invalid_profile_count']=0
        ref=base/'ground_truth.npz'
        if not ref.exists():ref=base/'baseline_reference.npz'
        truth=read_arrays(ref)
        if any(invalid_profiles(a) for a in truth.values()):raise ValueError('Invalid evaluator reference profiles')
        estimate=read_arrays(prediction)
        metrics={}
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for k in COUNTS:
                for name,v in compare(estimate[k],truth[k]).items():
                    if not np.isfinite(v):raise ValueError('Nonfinite derived metric')
                    metrics[k+'_'+name]=v
        thresholds={k+'_'+name:dict(comparison='<=',boundary=bound) for k in COUNTS for name,bound in bounds.items()}
        passed=all(v<=thresholds[k]['boundary'] for k,v in metrics.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as error:
        report['errors']=[f'{type(error).__name__}: {error}'];return report,2
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    report,code=evaluate(a.prediction)
    try:
        destination=Path(a.report);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as error:
        print(f'Cannot write report: {error}',file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
