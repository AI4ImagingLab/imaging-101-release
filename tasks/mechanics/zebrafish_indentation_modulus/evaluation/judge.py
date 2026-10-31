"""Standalone deterministic submission evaluator; no solver imports."""
from pathlib import Path
import argparse, json, sys
import numpy as np
ASSETS=Path(__file__).resolve().parent
KEYS=('reconstruction','contact_position')

def read_arrays(path):
    with np.load(path,allow_pickle=False) as z:
        out={}
        for key in KEYS:
            if key not in z: raise ValueError(f'Missing required key: {key}')
            a=z[key]
            if a.shape!=(1,454): raise ValueError(f'{key}: expected shape (1, 454), got {a.shape}')
            if a.dtype.kind!='f' or a.dtype.itemsize not in (4,8): raise ValueError(f'{key}: expected real float32 or float64')
            if not np.isfinite(a).all(): raise ValueError(f'{key}: contains nonfinite values')
            if key=='reconstruction' and np.any(a<0): raise ValueError('reconstruction: modulus must be nonnegative')
            out[key]=a.astype(np.float64)
    return out

def metrics(pred,ref):
    x=pred['reconstruction'].ravel(); y=ref['reconstruction'].ravel()
    norm=float(np.linalg.norm(x)*np.linalg.norm(y)); span=float(np.ptp(y))
    if span<=0 or np.linalg.norm(y)==0: raise ValueError('Degenerate modulus reference')
    return {'ncc':float(x@y/norm) if norm else 0.,'nrmse':float(np.sqrt(np.mean((x-y)**2))/span),'modulus_max_abs_pa':float(np.max(np.abs(x-y))),'contact_max_abs_m':float(np.max(np.abs(pred['contact_position']-ref['contact_position'])))}

def curve_loss(modulus,contact,force,height,segment,spring):
    keep=segment==0;f=force[keep].astype(float);z=height[keep].astype(float)+f/spring
    if not len(f) or np.ptp(f)<=0:raise ValueError('Degenerate measured approach')
    if not z.min()<=contact<=z.max():raise ValueError('Contact position outside measured approach range')
    depth=np.maximum(contact-z,0.);r=18.64e-6;nu=.5;q=depth/r
    model=(4/3)*modulus/(1-nu**2)*np.sqrt(r)*depth**1.5*(1-q/10-q*q/840+11*q**3/15120+1357*q**4/6652800)
    weight=np.minimum(np.abs(z-contact)/(2e-6),1.)
    denominator=np.sum(weight**2)
    if denominator<=0:raise ValueError('Degenerate contact weights')
    baseline=np.sum(weight**2*(f-model))/denominator
    return float(np.mean((weight*(f-baseline-model))**2)/np.ptp(f)**2)


def evaluate(prediction,assets=ASSETS):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((assets/'metrics.json').read_text(encoding='utf8'));boundary=cfg['maximum_curve_loss_excess_boundary']
        if type(boundary) not in (int,float) or not np.isfinite(boundary) or boundary<0:raise ValueError('Invalid metric boundary')
        ref=read_arrays(assets/'baseline_reference.npz');pred=read_arrays(prediction)
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as raw:
            force=raw['force'][0];height=raw['height'][0];segment=raw['segment'][0];offsets=raw['offsets'][0];spring=raw['spring_constant'][0]
        if offsets.shape!=(455,) or spring.shape!=(454,) or not np.all(spring>0):raise ValueError('Invalid acquisition calibration')
        excess=[]
        for i in range(454):
            lo,hi=offsets[i:i+2];inputs=force[lo:hi],height[lo:hi],segment[lo:hi],spring[i]
            old=curve_loss(ref['reconstruction'][0,i],ref['contact_position'][0,i],*inputs)
            new=curve_loss(pred['reconstruction'][0,i],pred['contact_position'][0,i],*inputs)
            excess.append(new-old)
        value=float(max(excess))
        if not np.isfinite(value):raise ValueError('Nonfinite evaluation metric')
        passed=value<=boundary
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics={'maximum_curve_loss_excess':value},thresholds={'maximum_curve_loss_excess':dict(comparison='<=',boundary=boundary)})
        return report,0 if passed else 1
    except Exception as exc:
        report['errors'].append(str(exc));return report,2

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
