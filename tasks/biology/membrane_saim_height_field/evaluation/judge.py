"""Standalone evaluation of all three declared fields, without solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
KEYS=('reconstruction','amplitude','background')
def score(estimate,reference):
    e=estimate.ravel();r=reference.ravel();scale=float(np.ptp(r));norm=float(np.linalg.norm(e)*np.linalg.norm(r))
    if scale<=0 or np.linalg.norm(r)==0:raise ValueError('Degenerate evaluator reference')
    return dict(ncc=float(e@r/norm) if norm else 0.,nrmse=float(np.sqrt(np.mean((e-r)**2))/scale))
def evaluate(prediction):
    result=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        configuration=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        with np.load(HERE/'baseline_reference.npz',allow_pickle=False) as f:ref={k:f[k] for k in KEYS}
        with np.load(prediction,allow_pickle=False) as f:
            if set(f.files)!=set(KEYS):raise ValueError('Submission must contain exactly reconstruction,amplitude,background')
            out={k:f[k] for k in KEYS}
        for key in KEYS:
            a=out[key]
            if a.shape!=(1,23,26) or a.dtype!=np.dtype('float64') or not np.isfinite(a).all():raise ValueError(f'{key}: expected finite real float64(1,23,26)')
            upper=200. if key=='reconstruction' else 10000.
            if np.any(a<0) or np.any(a>upper):raise ValueError(f'{key}: outside declared physical output interval')
            if ref[key].shape!=a.shape or ref[key].dtype!=np.float64 or not np.isfinite(ref[key]).all():raise ValueError('Invalid evaluator reference')
        measured={};thresholds={}
        for key in KEYS:
            for metric,value in score(out[key],ref[key]).items():
                name=key+'_'+metric;boundary=configuration[name+'_boundary'];direction='>=' if metric=='ncc' else '<='
                if not np.isfinite(boundary):raise ValueError('Nonfinite threshold')
                measured[name]=value;thresholds[name]=dict(comparison=direction,boundary=boundary)
        # Independently evaluate the known TE acquisition formula, not solver code.
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as f:y=f['intensities'];theta=f['angles'].ravel()
        ox=np.arcsin(1.33*np.sin(theta)/1.4719);si=np.arcsin(1.4719*np.sin(ox)/(3.8660+.017933j))
        delta=2*np.pi*1.4719/642*1924.9*np.cos(ox);p=1.4719*np.cos(ox);q=(3.8660+.017933j)*np.cos(si);b=1.33*np.cos(theta)
        u=np.cos(delta)-1j*np.sin(delta)/p*q;v=-1j*p*np.sin(delta)+np.cos(delta)*q;r=(u*b-v)/(u*b+v)
        phase=(4*np.pi*1.33/642*np.cos(theta))[:,None,None]*out['reconstruction'][0]
        predicted=out['amplitude'][0]*(1+2*r.real[:,None,None]*np.cos(phase)-2*r.imag[:,None,None]*np.sin(phase)+np.abs(r[:,None,None])**2)+out['background'][0]
        measured['observation_rms_adu']=float(np.sqrt(np.mean((predicted-y[0])**2)))
        thresholds['observation_rms_adu']=dict(comparison='<=',boundary=configuration['observation_rms_adu_boundary'])
        passed=all(measured[k]>=t['boundary'] if t['comparison']=='>=' else measured[k]<=t['boundary'] for k,t in thresholds.items())
        result.update(status='pass' if passed else 'fail',passed=passed,metrics=measured,thresholds=thresholds)
    except Exception as error:result['errors']=[f'{type(error).__name__}: {error}']
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    result=evaluate(Path(args.prediction))
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as error:print(f'Report write failed: {error}',file=sys.stderr);return 2
    print(json.dumps(result,allow_nan=False));return 0 if result['passed'] else 1 if result['status']=='fail' else 2
if __name__=='__main__':raise SystemExit(main())
