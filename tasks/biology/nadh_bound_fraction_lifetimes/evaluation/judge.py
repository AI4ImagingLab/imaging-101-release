"""Independent strict evaluator; no solver or domain-library imports."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,2,512,512),'bound_fraction':(1,512,512)}

def validate(path):
    with np.load(path,allow_pickle=False) as z:
        out={}
        for key,shape in SHAPES.items():
            if key not in z: raise ValueError('Missing required key: '+key)
            a=z[key]
            if a.shape!=shape: raise ValueError('Wrong shape: '+key)
            if a.dtype.kind!='f' or a.dtype.itemsize not in (4,8): raise ValueError('Expected float32 or float64: '+key)
            if not np.isfinite(a).all(): raise ValueError('Nonfinite values: '+key)
            out[key]=a
        return out

def normalized_error(a,b):
    scale=float(np.ptp(b))
    if scale<=0: raise ValueError('Degenerate reference range')
    return float(np.sqrt(np.mean((a-b)**2))/scale)

def evaluate(prediction,assets=ROOT):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        p=validate(prediction)
        refpath=assets/'ground_truth.npz'
        if not refpath.exists(): refpath=assets/'baseline_reference.npz'
        r=validate(refpath)
        rules=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as z:
            signal=z['measurements'][0];cal=z['calibration'][0]
        support=signal.sum(-1,dtype=np.float64)>0
        t=p['reconstruction'][0];f=p['bound_fraction'][0]
        if (t[:,support]<=0).any() or (t[0,support]>t[1,support]).any(): raise ValueError('Lifetimes must be positive and ordered on observed support')
        if ((f<0)|(f>1)).any(): raise ValueError('Fractions must lie in [0,1]')
        if np.any(t[:,~support]!=0) or np.any(f[~support]!=0): raise ValueError('Unobserved background must be zero')
        metrics={}
        omega=2*np.pi*80e-3;harmonics=np.arange(1,3)
        total=signal.sum(-1,dtype=np.float64)
        z=np.moveaxis(np.fft.rfft(signal,axis=-1)[...,harmonics].conj(),-1,0)[:,support]/total[support]
        pooled=cal.sum((0,1),dtype=np.float64)
        measured=np.fft.rfft(pooled)[harmonics].conj()/pooled.sum()
        z*=((1/(1-1j*omega*harmonics*3.05))/measured)[:,None]
        predicted=(1-f[support])[None]/(1-1j*omega*harmonics[:,None]*t[0,support])+f[support][None]/(1-1j*omega*harmonics[:,None]*t[1,support])
        metrics['forward_rms']=float(np.sqrt(np.mean(abs(predicted-z)**2)))
        def geometry(tau):
            points=1/(1-1j*omega*harmonics[:,None,None]*tau[None])
            direction=points[:,1]-points[:,0];offset=z-points[:,0]
            length=abs(direction)**2
            projection=np.real(offset*direction.conj())/np.maximum(length,1e-30)
            perpendicular=abs(offset-projection*direction)
            return projection,perpendicular,length
        projection,perpendicular,length=geometry(t[:,support])
        _,ref_perpendicular,_=geometry(r['reconstruction'][0][:,support])
        grid=.20+.01*np.arange(61)
        membership=np.full(support.sum(),np.inf)
        for candidate in grid:
            membership=np.minimum(membership,np.minimum(abs(t[0,support]-candidate),abs(t[1,support]-candidate)))
        metrics['grid_membership_max_ns']=float(membership.max())
        metrics['first_harmonic_line_max']=float(perpendicular[0].max())
        identifiable=length[1]>1e-20
        metrics['fraction_projection_max']=float(np.max(abs(f[support][identifiable]-projection[1,identifiable]))) if identifiable.any() else 0.
        metrics['second_harmonic_objective_excess_max']=float(np.maximum(perpendicular[1]**2-ref_perpendicular[1]**2,0).max())
        thresholds={key:dict(comparison='<=',boundary=rules[key+'_boundary']) for key in metrics}
        if any(not isinstance(v['boundary'],(int,float)) or not np.isfinite(v['boundary']) or v['boundary']<0 for v in thresholds.values()): raise ValueError('Invalid metric boundary')
        if not all(np.isfinite(list(metrics.values()))): raise ValueError('Undefined metric')
        report.update(metrics=metrics,thresholds=thresholds)
        report['passed']=all(v<=thresholds[k]['boundary'] for k,v in metrics.items())
        report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as exc:
        report['errors']=[str(exc)]
        return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();report,code=evaluate(Path(args.prediction))
    try:
        dest=Path(args.report);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__': sys.exit(main())
