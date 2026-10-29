"""Standalone deterministic Judge; never imports solver or submitted Python."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,2,5),'instrument_offset':(1,3),'instrument_jitter':(1,3)}

def load_arrays(path):
    if not Path(path).is_file():raise ValueError('missing file: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        out={}
        for k,shape in SHAPES.items():
            if k not in z:raise ValueError('missing key: '+k)
            a=z[k]
            if a.shape!=shape:raise ValueError('shape: '+k)
            if a.dtype!=np.float64:raise ValueError('dtype must be real float64: '+k)
            if not np.isfinite(a).all():raise ValueError('nonfinite: '+k)
            out[k]=a.copy()
    return out

def metrics(pred,ref):
    a=pred['reconstruction'][0];b=ref['reconstruction'][0]
    difference=np.abs(a-b)
    difference[:,1]=np.abs((a[:,1]-b[:,1]+b[:,0]/2)%b[:,0]-b[:,0]/2)
    difference[:,3]=np.abs(np.angle(np.exp(1j*(a[:,3]-b[:,3]))))
    m={n:float(difference[:,j].max()) for j,n in enumerate(['period_abs_day','epoch_phase_abs_day','eccentricity_abs','omega_phase_abs_rad','amplitude_abs_mps'])}
    for key,name in [('instrument_offset','offset_abs_mps'),('instrument_jitter','jitter_abs_mps')]:m[name]=float(np.abs(pred[key]-ref[key]).max())
    return m

def observation_metrics(pred,raw):
    t=raw['time'][0];inst=raw['instrument'][0];model=np.zeros(t.shape)
    orbits=pred['reconstruction'][0];gamma=pred['instrument_offset'][0];jitter=pred['instrument_jitter'][0]
    if np.any(orbits[:,0]<=0) or np.any(orbits[:,2]<0) or np.any(orbits[:,2]>.99) or np.any(orbits[:,4]<=0) or np.any(jitter<=0):raise ValueError('nonphysical orbital/noise parameters')
    # Independent Newton solve rather than the solver's third-order corrections.
    for P,tc,e,w,K in orbits:
        f=np.pi/2-w;Ec=2*np.arctan(np.tan(f/2)*np.sqrt((1-e)/(1+e)))
        tp=tc-P*(Ec-e*np.sin(Ec))/(2*np.pi)
        M=2*np.pi*np.remainder((t-tp)/P,1);E=M.copy()
        for _ in range(100):
            step=(E-e*np.sin(E)-M)/(1-e*np.cos(E));E-=step
            if np.max(np.abs(step))<1e-13:break
        nu=2*np.arctan2(np.sqrt(1+e)*np.sin(E/2),np.sqrt(1-e)*np.cos(E/2))
        model+=K*(np.cos(nu+w)+e*np.cos(w))
    model+=gamma[inst]
    var=raw['uncertainty'][0]**2+jitter[inst]**2
    res=raw['velocity'][0]-model
    return {'weighted_residual_rms':float(np.sqrt(np.mean(res**2/var)))}

def judge(prediction,assets=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        pred=load_arrays(prediction)
        refpath=assets/'ground_truth.npz'
        if not refpath.is_file():refpath=assets/'baseline_reference.npz'
        ref=load_arrays(refpath)
        cfg=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        bounds={k[:-9]:v for k,v in cfg.items() if k.endswith('_boundary')}
        if cfg.get('comparisons')!={k:'<=' for k in bounds}:raise ValueError('invalid metric comparisons')
        if not bounds or any(not np.isfinite(v) or v<0 for v in bounds.values()):raise ValueError('invalid metric configuration')
        m=metrics(pred,ref)
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as raw:m.update(observation_metrics(pred,raw))
        if set(m)!=set(bounds):raise ValueError('metric configuration mismatch')
        if not all(np.isfinite(v) for v in m.values()):raise ValueError('undefined metric')
        report['metrics']=m;report['thresholds']={k:{'comparison':'<=','boundary':v} for k,v in bounds.items()}
        report['passed']=all(m[k]<=v for k,v in bounds.items());report['status']='pass' if report['passed'] else 'fail'
    except Exception as e:report['errors']=[str(e)]
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);a=parser.parse_args()
    r=judge(a.prediction)
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('report write failed: '+str(e),file=sys.stderr);return 2
    return 0 if r['passed'] else 1 if r['status']=='fail' else 2
if __name__=='__main__':raise SystemExit(main())
