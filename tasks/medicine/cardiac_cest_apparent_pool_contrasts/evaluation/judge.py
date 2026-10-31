"""Standalone evaluator: no solver imports or submitted code execution."""
import argparse,json,sys
from pathlib import Path
import numpy as np

SHAPES={'amplitude':(1,192,192,6),'width':(1,192,192,6),'center':(1,192,192,6),'field_shift':(1,192,192)}
NAMES=('water','mt','amide','creatine','noe35','noe16')
BOUNDS={'amplitude':([.02,0,0,0,0,0],[1,.5,.3,.5,.25,.25]),'width':([.01,30,.5,.5,.5,.5],[10,60,5,5,5,5]),'center':([-1e-6,-2.5,3.2,1.6,-4,-1.8],[1e-6,0,4,2.6,-1.5,-1.2])}

def load_arrays(path):
    with np.load(path,allow_pickle=False) as z:
        out={}
        for key,shape in SHAPES.items():
            if key not in z:raise ValueError('missing required key: '+key)
            v=z[key]
            if v.shape!=shape:raise ValueError('wrong shape: '+key)
            if v.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('expected real float32 or float64: '+key)
            if not np.all(np.isfinite(v)):raise ValueError('nonfinite values: '+key)
            out[key]=v
    return out

def normalized_observations(raw):
    im=raw['measurements'][0];offsets=raw['offsets'][0];refs=np.where(offsets>15)[0];keep=np.where(offsets<=15)[0]
    upper=np.searchsorted(refs,keep);lo=refs[upper-1];hi=refs[upper];fraction=(keep-lo)/(hi-lo)
    reference=im[:,:,lo]*(1-fraction)+im[:,:,hi]*fraction
    with np.errstate(divide='ignore',invalid='ignore'):z=np.nan_to_num(im[:,:,keep]/reference)
    return z,offsets[keep],raw['observation_mask'][0]

def observation_mse(arrays,spectra,offsets,mask):
    x=offsets[None,:]-arrays['field_shift'][0][mask,None]
    predicted=np.ones_like(x)
    for k in range(6):
        a=arrays['amplitude'][0,...,k][mask,None];w=arrays['width'][0,...,k][mask,None];c=arrays['center'][0,...,k][mask,None]
        predicted-=a*.25*w*w/(.25*w*w+(x-c)**2)
    return np.mean((predicted-spectra[mask])**2,axis=1)


def observation_rmse(arrays,spectra,offsets,mask):
    return float(np.sqrt(np.mean(observation_mse(arrays,spectra,offsets,mask))))

def evaluate(prediction,assets=None):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    base=Path(assets) if assets is not None else Path(__file__).resolve().parent
    try:
        p=load_arrays(prediction)
        refpath=base/'ground_truth.npz'
        if not refpath.exists():refpath=base/'baseline_reference.npz'
        ref=load_arrays(refpath);config=json.loads((base/'metrics.json').read_text(encoding='utf-8'))
        with np.load(base.parent/'data/raw_data.npz',allow_pickle=False) as raw:z,off,mask=normalized_observations(raw)
        if mask.shape!=(192,192) or not np.any(mask):raise ValueError('invalid evaluator mask')
        for key,v in p.items():
            if np.any(v[0][~mask]!=0):raise ValueError('outside-support values must be zero: '+key)
            if key in BOUNDS:
                lo,hi=map(np.array,BOUNDS[key]);inside=v[0][mask]
                if np.any(inside<lo-1e-7) or np.any(inside>hi+1e-7):raise ValueError('physical parameter range violated: '+key)
            elif np.any(np.abs(v[0][mask])>10+1e-7):raise ValueError('field shift exceeds declared range')
        error=observation_mse(p,z,off,mask);attained=observation_mse(ref,z,off,mask)
        metrics={'maximum_voxel_mse_excess':float(np.max(error-attained)),'observation_rmse':float(np.sqrt(np.mean(error)))}
        thresholds={key:dict(comparison='<=',boundary=config[key+'_boundary']) for key in metrics}
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('undefined metric')
        if not all(np.isfinite(q['boundary']) and q['boundary']>0 for q in thresholds.values()):raise ValueError('invalid threshold configuration')
        passed=all(metrics[k]<=q['boundary'] for k,q in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)];return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:print('report write failed: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
