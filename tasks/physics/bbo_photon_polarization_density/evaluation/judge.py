"""Standalone evaluator: no solver imports or output transformations."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
def validate(path):
    if not Path(path).is_file(): raise ValueError('Missing prediction or reference file')
    with np.load(path,allow_pickle=False) as z:
        if not {'reconstruction','intensity'}<=set(z.files): raise ValueError('Missing required keys')
        r=z['reconstruction'];s=z['intensity']
    if r.shape!=(1,4,4) or s.shape!=(1,1): raise ValueError('Wrong declared shape')
    if r.dtype not in (np.dtype('complex64'),np.dtype('complex128')) or s.dtype not in (np.dtype('float32'),np.dtype('float64')): raise ValueError('Wrong declared dtype')
    if not np.isfinite(r).all() or not np.isfinite(s).all(): raise ValueError('Nonfinite prediction')
    return r[0],float(s[0,0])
def evaluate(path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        r,s=validate(path)
        reference=HERE/'ground_truth.npz' if (HERE/'ground_truth.npz').is_file() else HERE/'baseline_reference.npz'
        ref,sref=validate(reference)
        if abs(np.trace(ref)-1)>1e-7 or np.linalg.norm(ref-ref.conj().T)>1e-7 or np.linalg.eigvalsh((ref+ref.conj().T)/2).min() < -1e-7:
            raise ValueError('Invalid hidden reference physicality')
        cfg=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        names=['density_distance','intensity_relative_error','observation_rmse','physicality']
        boundaries=[float(cfg[n+'_boundary']) for n in names]
        if not np.isfinite(boundaries).all() or min(boundaries)<=0: raise ValueError('Invalid metric configuration')
        if sref<=0: raise ValueError('Invalid hidden reference normalization')
        report['thresholds']={n:{'comparison':'<=','boundary':b} for n,b in zip(names,boundaries)}
        if s<=0:
            report.update(status='fail',errors=['Intensity must be positive']);return report,1
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z: c=z['counts'][0];p=z['projectors'][0]
        if c.shape!=(36,) or p.shape!=(36,4,4) or not np.isfinite(c).all() or not np.isfinite(p).all(): raise ValueError('Invalid public calibration')
        pred=np.array([s*np.real(np.trace(a@r)) for a in p])
        physical=max(float(abs(np.trace(r)-1)),float(np.linalg.norm(r-r.conj().T)),max(0.,float(-np.linalg.eigvalsh((r+r.conj().T)/2).min())))
        values=[float(np.linalg.norm(r-ref)),abs(s-sref)/sref,float(np.sqrt(np.mean((pred-c)**2))/s),physical]
        if not np.isfinite(values).all(): raise ValueError('Nonfinite metric')
        report['metrics']=dict(zip(names,values)); passed=all(v<=b for v,b in zip(values,boundaries))
        report.update(status='pass' if passed else 'fail',passed=passed)
        return report,0 if passed else 1
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},errors=[str(exc)])
        return report,2
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc: print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__': sys.exit(main())
