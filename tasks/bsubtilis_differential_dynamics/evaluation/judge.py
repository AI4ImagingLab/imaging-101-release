"""Standalone strict DDM parameter and direct-observation evaluator."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.fft import rfft2,fftshift
HOME=Path(__file__).resolve().parent
LAGS=np.array([1,2,5,10,20,50,100,200,500])

def direct_observation(frames):
    """Independent time-difference DDM; no autocorrelation inversion imports."""
    h,w=128,128;half=65;pixel=75.59/128
    kx=2*np.pi*np.fft.fftfreq(w)[:half];kx*=1/pixel
    ky=2*np.pi*np.fft.fftshift(np.fft.fftfreq(h));ky*=1/pixel
    xx,yy=np.meshgrid(kx,ky);radius=np.hypot(xx,yy)
    edges=np.linspace(0,kx[1]*90,90);bins=np.searchsorted(edges,radius)
    valid=(bins<90);valid[:,0]=False;valid[64,:]=False
    mult=np.full(radius.shape,2.);mult[:,0]=1;mult[:,-1]=1
    ids=bins[valid];weights=mult[valid];tot=np.bincount(ids,weights,minlength=90)
    q=np.divide(np.bincount(ids,radius[valid]*weights,minlength=90),tot,out=np.zeros(90),where=tot>0)
    selected=(q>=.75)&(q<=4);observed=[]
    for movie in frames[0]:
        ft=fftshift(rfft2(movie.astype(np.float64),workers=1)/np.sqrt(h*w),axes=-2)
        curves=[]
        for lag in LAGS:
            # Chunk over Fourier modes to avoid large temporary arrays.
            flat=ft.reshape(len(movie),-1);power=np.empty(flat.shape[1])
            for a in range(0,len(power),256):
                b=min(a+256,len(power));power[a:b]=np.mean(np.abs(flat[lag:,a:b]-flat[:-lag,a:b])**2,axis=0)
            radial=np.divide(np.bincount(ids,power.reshape(h,half)[valid]*weights,minlength=90),tot,out=np.zeros(90),where=tot>0)
            curves.append(radial[selected])
        observed.append(np.array(curves).T)
    return np.array(observed),q[selected]

def validate(z):
    for key,shape in [('reconstruction',(1,2,39,4)),('speed',(1,2))]:
        if key not in z:raise ValueError('Missing required key '+key)
        a=z[key]
        if a.shape!=shape or a.dtype!=np.dtype('float64'):raise ValueError(key+' must have shape '+str(shape)+' and real float64 dtype')
        if not np.isfinite(a).all():raise ValueError(key+' contains NaN or infinity')
    p=z['reconstruction']
    if np.any(p[...,[0,2]]<=0) or np.any(p[...,3]<1) or np.any(z['speed']<=0):raise ValueError('A, Gamma, speed must be positive and beta >=1')

def score(pred,ref,observed,q):
    measured={};p=pred['reconstruction'][0];truth=ref['reconstruction'][0]
    for c in range(2):
        for j,name in enumerate(('amplitude','background','relaxation','exponent')):
            denominator=np.linalg.norm(truth[c,:,j])
            if denominator==0:raise ValueError('Degenerate hidden reference '+name)
            measured[f'condition{c}_{name}_relative_l2']=float(np.linalg.norm(p[c,:,j]-truth[c,:,j])/denominator)
        measured[f'condition{c}_speed_relative_error']=float(abs(pred['speed'][0,c]/ref['speed'][0,c]-1))
        A,B,G,beta=p[c].T;tau=LAGS*.01
        fitted=A[:,None]*(1-np.exp(-(G[:,None]*tau)**beta[:,None]))+B[:,None]
        measured[f'condition{c}_observation_relative_l2']=float(np.linalg.norm(fitted-observed[c])/np.linalg.norm(observed[c]))
    return measured

def evaluate(prediction,report):
    result={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]};code=2
    try:
        with np.load(prediction,allow_pickle=False) as z:pred={k:z[k] for k in z.files}
        validate(pred)
        config=json.loads((HOME/'metrics.json').read_text(encoding='utf-8'))
        # Actual truth has precedence when present; this real experiment has no truth.
        reference=HOME/('ground_truth.npz' if (HOME/'ground_truth.npz').exists() else 'baseline_reference.npz')
        with np.load(reference,allow_pickle=False) as z:ref={k:z[k] for k in z.files}
        validate(ref)
        with np.load(HOME.parent/'data/raw_data.npz',allow_pickle=False) as z:frames=z['frames']
        if frames.shape!=(1,2,2000,128,128) or frames.dtype!=np.uint8:raise ValueError('Invalid public observation calibration')
        observed,q=direct_observation(frames)
        metrics=score(pred,ref,observed,q)
        thresholds={k:{'comparison':'<=','boundary':config['observation_relative_l2_boundary'] if 'observation' in k else config['speed_relative_error_boundary'] if 'speed' in k else config['parameter_relative_l2_boundary']} for k in metrics}
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Undefined metric')
        passed=all(v<=thresholds[k]['boundary'] for k,v in metrics.items())
        result.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=thresholds);code=0 if passed else 1
    except Exception as exc:result['errors']=[f'{type(exc).__name__}: {exc}']
    try:
        target=Path(report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prediction',required=True);ap.add_argument('--report',required=True);a=ap.parse_args();sys.exit(evaluate(a.prediction,a.report))
