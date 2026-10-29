"""Standalone independent evaluator; never imports reference or submitted solver."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.special import erf

KEYS=('reconstruction','molecule_number','correlation_offset')
SHAPE=(1,6,40)

def vector_metrics(estimate,reference):
    """Uncentered cosine NCC and reference dynamic-range NRMSE."""
    estimate=np.asarray(estimate).ravel();reference=np.asarray(reference).ravel()
    scale=float(np.ptp(reference)); norm=float(np.linalg.norm(reference));enorm=float(np.linalg.norm(estimate))
    if scale<=0 or norm<=0:raise ValueError('Degenerate reference metric configuration')
    return {'ncc':float(np.dot(estimate,reference)/(enorm*norm)) if enorm else 0.,'nrmse':float(np.sqrt(np.mean((estimate-reference)**2))/scale)}

def evaluate(prediction,asset_dir=None):
    """Return strict JSON-compatible report and CLI exit code."""
    root=Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        with np.load(root/'baseline_reference.npz',allow_pickle=False) as a:
            refs={key:a[key] for key in KEYS};mask=a['valid_mask'];observed=a['observed_correlation'];lag=a['lag_times']
        truth_path=root/'ground_truth.npz'
        if truth_path.is_file():
            with np.load(truth_path,allow_pickle=False) as a:refs={key:a[key] for key in KEYS}
        if mask.dtype!=np.bool_ or mask.shape!=SHAPE[1:] or int(mask.sum())!=52:raise ValueError('Invalid fixed evaluation ROI')
        if observed.shape!=(52,104) or lag.shape!=(104,) or not np.isfinite(observed).all() or not np.isfinite(lag).all() or np.any(observed[:,0]==0):raise ValueError('Invalid derived observation assets')
        for key,ref in refs.items():
            if ref.shape!=SHAPE or ref.dtype!=np.float64 or not np.isfinite(ref).all():raise ValueError('Invalid reference '+key)
            vector_metrics(ref[0][mask],ref[0][mask])
        definition=json.loads((root/'metrics.json').read_text(encoding='utf8'))
        thresholds=definition['thresholds']
        expected={key+'_'+metric for key in KEYS for metric in ('ncc','nrmse')}|{'correlation_residual'}
        if set(thresholds)!=expected:raise ValueError('Undefined metric configuration')
        for key,value in thresholds.items():
            if set(value)!= {'comparison','boundary'} or value['comparison'] not in ('>=','<=') or not isinstance(value['boundary'],(int,float)) or not np.isfinite(value['boundary']):raise ValueError('Invalid threshold '+key)
            if value['comparison']!=('>=' if key.endswith('_ncc') else '<='):raise ValueError('Wrong comparison '+key)
        report['thresholds']=thresholds
        with np.load(prediction,allow_pickle=False) as a:
            if set(a.files)!=set(KEYS):raise ValueError('Prediction must contain exactly: '+', '.join(KEYS))
            data={key:a[key] for key in KEYS}
        for key,value in data.items():
            if value.shape!=SHAPE or value.dtype!=np.float64:raise ValueError('Expected real float64 shape(1,6,40): '+key)
            if not np.isfinite(value).all():raise ValueError('NaN or infinity: '+key)
            if np.any(value[0][~mask]!=0):raise ValueError('Nonzero values outside public cell bins: '+key)
        # Negative/zero physical estimates are valid-shaped poor submissions,
        # not evaluator errors. Avoid undefined model arithmetic explicitly.
        if np.any(data['reconstruction'][0][mask]<=0) or np.any(data['molecule_number'][0][mask]<=0):
            report.update(status='fail',errors=['Positive diffusion and molecule number required in every cell bin']);return report,1
        measured={}
        for key in KEYS:
            for metric,value in vector_metrics(data[key][0][mask],refs[key][0][mask]).items():measured[key+'_'+metric]=value
        D=data['reconstruction'][0][mask,None];N=data['molecule_number'][0][mask,None];offset=data['correlation_offset'][0][mask,None]
        k=.32/(2*np.sqrt(.19**2+2.1*D*lag[None,:]))
        overlap=erf(k)+(np.exp(-k*k)-1)/(k*np.sqrt(np.pi))
        model=overlap*overlap/N+offset
        measured['correlation_residual']=float(np.sqrt(np.mean(((model-observed)/observed[:,0,None])**2)))
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('Undefined measured metric')
        passed=all(value>=thresholds[key]['boundary'] if thresholds[key]['comparison']=='>=' else value<=thresholds[key]['boundary'] for key,value in measured.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=measured);return report,0 if passed else 1
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},errors=[str(exc)]);return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Could not write evaluation report: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
