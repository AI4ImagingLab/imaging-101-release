"""Standalone deterministic evaluator; imports no solver/submission code."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.linalg import solve

ASSET_DIR=Path(__file__).resolve().parent
SHAPES={'slip_rate':(1,1299),'basin_coefficient':(1,1),'datum_offset':(1,5)}
GROUPS={'deep_slip':slice(1292,1297),'east_slip':slice(1297,1298),
        'west_slip':slice(1298,1299),'basin_coefficient':slice(1299,1300)}
GROUPS.update({'datum'+str(j):slice(1300+j,1301+j) for j in range(5)})


def load_output(path):
    """Validate every declared field with no reshape/cast/crop/fallback."""
    path=Path(path)
    if not path.is_file():raise ValueError('Missing NPZ file: '+str(path))
    result={}
    with np.load(path,allow_pickle=False) as z:
        for key,shape in SHAPES.items():
            if key not in z.files:raise ValueError('Missing required key: '+key)
            a=z[key]
            if a.shape!=shape:raise ValueError('Invalid shape for '+key+': '+str(a.shape))
            if a.dtype.kind!='f' or a.dtype.itemsize!=8:raise ValueError('Expected float64 for '+key)
            if not np.isfinite(a).all():raise ValueError('Nonfinite values in '+key)
            result[key]=a.copy()
    return result


def load_reference(directory=ASSET_DIR):
    """Actual truth takes precedence; malformed truth never falls back."""
    directory=Path(directory)
    truth=directory/'ground_truth.npz'
    selected=truth if truth.exists() else directory/'baseline_reference.npz'
    return load_output(selected)


def flatten_output(arrays):
    return np.concatenate((arrays['slip_rate'][0],arrays['basin_coefficient'][0],arrays['datum_offset'][0]))


def field_metrics(prediction,reference):
    p=flatten_output(prediction);r=flatten_output(reference)
    a=p[:1292];b=r[:1292];span=np.ptp(b);norm=np.linalg.norm(b)
    if span<=0 or norm<=0:raise ValueError('Degenerate main-slip reference')
    pn=np.linalg.norm(a)
    measured={'main_slip_ncc':float(a@b/(pn*norm)) if pn>0 else 0.,
              'main_slip_nrmse':float(np.sqrt(np.mean((a-b)**2))/span)}
    for name,s in GROUPS.items():measured[name+'_maxabs']=float(np.max(abs(p[s]-r[s])))
    return measured


def evaluate(prediction_path,directory=ASSET_DIR):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        prediction=load_output(prediction_path)
        reference=load_reference(directory)
        directory=Path(directory)
        config=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
        thresholds={'main_slip_ncc':{'comparison':'>=','boundary':config['main_slip_ncc_boundary']},
                    'main_slip_nrmse':{'comparison':'<=','boundary':config['main_slip_nrmse_boundary']},
                    'observation_weighted_rms':{'comparison':'<=','boundary':config['observation_weighted_rms_boundary']}}
        for name in GROUPS:thresholds[name+'_maxabs']={'comparison':'<=','boundary':config['absolute_boundaries'][name]}
        for name,threshold in thresholds.items():
            boundary=threshold['boundary']
            if isinstance(boundary,bool) or not isinstance(boundary,(int,float)):
                raise ValueError('Metric boundary must be a real JSON number: '+name)
            if not np.isfinite(boundary):raise ValueError('Nonfinite metric boundary: '+name)
            if name=='main_slip_ncc':
                if not -1.<=boundary<=1.:raise ValueError('NCC boundary outside [-1,1]')
            elif boundary<0.:
                raise ValueError('Negative error boundary: '+name)
        with np.load(directory/'baseline_reference.npz',allow_pickle=False) as calibration:
            g=calibration['_green'];cd=calibration['_data_covariance'];d=calibration['_observations']
        for name,a,shape in [('operator',g,(3554,1305)),('noise covariance',cd,(3554,3554)),('observations',d,(3554,))]:
            if a.shape!=shape or a.dtype.kind!='f' or not np.isfinite(a).all():raise ValueError('Invalid hidden calibration '+name)
        measured=field_metrics(prediction,reference)
        residual=d-g@flatten_output(prediction)
        quadratic=float(residual@solve(cd,residual,assume_a='pos',check_finite=False))
        if quadratic<0:raise ValueError('Invalid noise covariance quadratic')
        measured['observation_weighted_rms']=float(np.sqrt(quadratic/len(d)))
        if any(not np.isfinite(v) for v in measured.values()):raise ValueError('Nonfinite measured metric')
        passed=all(measured[k]>=t['boundary'] if t['comparison']=='>=' else measured[k]<=t['boundary'] for k,t in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured,thresholds=thresholds)
    except Exception as exc:
        report['errors']=[type(exc).__name__+': '+str(exc)]
    return report


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args(argv);report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:
        print('Could not write report: '+str(exc),file=sys.stderr);return 2
    return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2


if __name__=='__main__':raise SystemExit(main())
