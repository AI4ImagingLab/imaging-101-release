"""Independent NPZ evaluator. Never imports a reconstruction implementation."""
from pathlib import Path
import argparse,json,sys
import numpy as np

HIDDEN=Path(__file__).resolve().parent
METRICS=HIDDEN/'metrics.json'
PUBLIC=HIDDEN.parent/'data/raw_data.npz'
KEYS=('reconstruction','components')

def load_arrays(path,label):
    if not Path(path).is_file():raise ValueError(f'{label} file unavailable: {path}')
    with np.load(path,allow_pickle=False) as f:
        arrays={}
        for key in KEYS:
            if key not in f:raise ValueError(f'{label}: missing required key {key}')
            x=f[key]
            if x.shape!=(1,401):raise ValueError(f'{label}: {key} shape must be (1,401)')
            if x.dtype!=np.complex128:raise ValueError(f'{label}: {key} dtype must be complex128')
            if not np.isfinite(x).all():raise ValueError(f'{label}: {key} contains nonfinite values')
            arrays[key]=x.copy()
    return arrays

def relative_l2(estimate,reference):
    denominator=np.linalg.norm(reference)
    if denominator==0 or not np.isfinite(denominator):raise ValueError('Reference norm is zero or undefined')
    result=float(np.linalg.norm(estimate-reference)/denominator)
    if not np.isfinite(result):raise ValueError('Nonfinite relative error from numerical overflow')
    return result

def observation_chi2(components):
    with np.load(PUBLIC,allow_pickle=False) as f:
        freq=f['frequency'];p=f['polarization'];sigma=f['noise']
        if any(x.shape!=(1,288) for x in (freq,p,sigma)):raise ValueError('Evaluator public-data shape error')
        if not all(np.isfinite(x).all() for x in (freq,p,sigma)) or np.any(freq<=0) or np.any(sigma<=0):
            raise ValueError('Evaluator public-data values invalid')
        lambda2=(299792458.0/freq[0])**2;w=1/sigma[0]**2
        lambda0=np.sum(w*lambda2)/np.sum(w);phi=np.arange(-600.,603.,3.)
        model=np.exp(2j*np.outer(lambda2-lambda0,phi))@components[0]
        result=float(np.sum(abs(model-p[0])**2/sigma[0]**2)/(2*len(lambda2)))
        if not np.isfinite(result):raise ValueError('Nonfinite observation metric')
        return result

def representation_metrics(arrays):
    with np.load(PUBLIC,allow_pickle=False) as f:
        t=(299792458./f['frequency'][0])**2;w=1/f['noise'][0]**2;p=f['polarization'][0]
    t0=np.sum(t*w)/np.sum(w);phi=np.arange(-600.,603.,3.)
    A=np.exp(2j*np.outer(t-t0,phi));dirty=A.conj().T@(w*p)/w.sum()
    c=arrays['components'][0]
    width=3.8/np.ptp(t)
    gaussian=np.exp(-4*np.log(2)*((phi[:,None]-phi[None,:])/width)**2)
    expected=gaussian@c+dirty-A.conj().T@(w*(A@c))/w.sum()
    return {'component_l1_flux':float(np.sum(abs(c))),'restoration_identity_relative_error':relative_l2(arrays['reconstruction'][0],expected)}

def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            settings=json.loads(METRICS.read_text(encoding='utf8'))
            for name in ('component_l1_flux','restoration_identity_relative_error','forward_reduced_chi2'):
                value=settings[name+'_boundary']
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
                    raise ValueError(f'Invalid metric boundary for {name}')
                report['thresholds'][name]={'comparison':'<=','boundary':value}
            arrays=load_arrays(prediction,'prediction')
            measured=representation_metrics(arrays)
            measured['forward_reduced_chi2']=observation_chi2(arrays['components'])
            report['metrics']=measured
            passed=all(value<=report['thresholds'][name]['boundary'] for name,value in measured.items())
            report.update(status='pass' if passed else 'fail',passed=bool(passed))
            return report,0 if passed else 1
    except Exception as exc:
        report['metrics']={};report['errors']=[str(exc)]
        return report,2

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args(argv);report,status=evaluate(Path(args.prediction))
    try:
        destination=Path(args.report);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return status

if __name__=='__main__':raise SystemExit(main())
