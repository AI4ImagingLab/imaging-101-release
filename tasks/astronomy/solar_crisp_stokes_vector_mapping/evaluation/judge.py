"""Standalone evaluator. Independent matrix-transfer and SciPy Voigt check."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import wofz
from scipy.signal import correlate

HERE=Path(__file__).resolve().parent
NAMES=('field','inclination','azimuth','velocity','doppler_width','opacity','damping','source_constant','source_gradient')

def finite_boundary(config,name,cosine=False):
    v=config[name]
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v):
        raise ValueError('invalid finite numeric boundary: '+name)
    if (cosine and not -1<=v<=1) or (not cosine and v<0):
        raise ValueError('boundary outside valid range: '+name)
    return v

def metrics(a,b):
    scale=np.ptp(b)
    if not scale>0: raise ValueError('constant evaluator reference')
    # Scaling first avoids intermediate overflow for valid finite submissions.
    norm=np.linalg.norm(a/max(np.max(np.abs(a)),1.))
    cosine=0. if norm==0 else float(np.dot((a/max(np.max(np.abs(a)),1.)).ravel(),b.ravel())/(norm*np.linalg.norm(b)))
    error=float(np.sqrt(np.mean(((a-b)/scale)**2)))
    if not np.isfinite([cosine,error]).all(): raise ValueError('nonfinite metric arithmetic')
    return cosine,error

def azimuth_metrics(a,b):
    difference=a-b
    delta=.5*np.arctan2(np.sin(2*difference),np.cos(2*difference))
    result=float(np.mean(np.cos(2*delta))),float(np.sqrt(np.mean(delta**2))/np.pi)
    if not np.isfinite(result).all():raise ValueError('nonfinite azimuth metric arithmetic')
    return result

def observation_rms(parameters,raw_path):
    """Independent K-matrix inverse with special-function Voigt and direct PSF.

    Parameters describe a linear-source, depth-independent atmosphere; source
    vector is S0*eI+mu*S1*K^-1*eI. No reference-solver imports or derivatives.
    """
    with np.load(raw_path,allow_pickle=False) as z:
        observed=z['stokes'][0]; wave=z['wavelength'][0]-6302.4931
        cavity=z['cavity_velocity'][0]; psfs=[z['spectral_psf_6301'][0],z['spectral_psf_6302'][0]]
    regions=[]; indices=[]
    for w,step in ((wave[:17],.005),(wave[17:],.015)):
        n=int(np.rint(w[-1]/step)-np.rint(w[0]/step)+11)
        fine=w[0]+(np.arange(n)-5)*step
        indices.append(np.abs(fine[:,None]-w).argmin(axis=0));regions.append(fine+6302.4931)
    m=parameters.reshape(-1,9).copy();m[:,3]-=cavity.ravel()+.45
    sigma=np.array([.003,.0003,.0003,.003/3.5]);squares=np.zeros(4)
    # Explicit fixed Zeeman transitions from upper/lower magnetic sublevels.
    patterns=[]
    fac=4.80320441e-10/(4e8*np.pi*9.10938188e-28*2.99792458e10**2)
    for center,j1,j2,g1,g2,gf in ((6301.4995,2,2,1.84,1.50,1.),(6302.4931,1,0,2.49,0.,10**(-.442))):
        lines=[]
        for upper in np.arange(j2,-j2-1,-1):
            for pol in range(3):
                lower=upper-1+pol
                if abs(lower)>j1: continue
                if j1==j2:
                    st=(1.5*(j1-lower)*(j1+lower+1),3*lower**2,1.5*(j1+lower)*(j1-lower+1))[pol]/(j1*(j1+1)*(2*j1+1))
                else:
                    st=(1.5*(j1-lower)*(j1-lower-1),3*(j1-lower)*(j1+lower),1.5*(j1+lower)*(j1+lower-1))[pol]/(j1*(2*j1-1)*(2*j1+1))
                if st>1e-5: lines.append((pol,st,-(g2*upper-g1*lower)*fac*center**2))
        patterns.append((center,gf,lines))
    for begin in range(0,len(m),32):
        p=m[begin:begin+32];B,theta,phi,v,width,eta,a,S0,S1=p.T
        wl=np.concatenate(regions)[None,:];shape=(len(p),wl.shape[1])
        absorption=np.zeros((*shape,4));absorption[...,0]=1
        dispersion=np.zeros((*shape,3));ct=np.cos(theta)[:,None];st=np.sin(theta)[:,None]**2
        ca=np.cos(2*phi)[:,None];sa=np.sin(2*phi)[:,None]
        for center,gf,lines in patterns:
            profile=np.zeros((*shape,3),complex)
            for pol,strength,split in lines:
                x=(wl-center-center*v[:,None]/299792.458-split*B[:,None])/width[:,None]
                profile[...,pol]+=strength*wofz(x+1j*a[:,None])
            for values,target,offset in ((profile.real,absorption,1),(profile.imag,dispersion,0)):
                delta=values[...,1]-(values[...,0]+values[...,2])/2
                target[...,offset]+=gf*eta[:,None]/2*delta*st*ca
                target[...,offset+1]+=gf*eta[:,None]/2*delta*st*sa
                target[...,offset+2]+=gf*eta[:,None]/2*(values[...,2]-values[...,0])*ct
            absorption[...,0]+=gf*eta[:,None]/2*(profile[...,1].real*st+(profile[...,0].real+profile[...,2].real)*(1+ct**2)/2)
        I,Q,U,V=absorption.transpose(2,0,1);rQ,rU,rV=dispersion.transpose(2,0,1)
        K=np.zeros((*shape,4,4));K[...,range(4),range(4)]=I[...,None]
        K[...,0,1:]=absorption[...,1:];K[...,1:,0]=absorption[...,1:]
        K[...,1,2]=rV;K[...,2,1]=-rV;K[...,1,3]=-rU;K[...,3,1]=rU;K[...,2,3]=rQ;K[...,3,2]=-rQ
        rhs=np.zeros((*shape,4));rhs[...,0]=1
        spectrum=(np.linalg.solve(K,rhs[...,None])[...,0]*.93*S1[:,None,None]);spectrum[...,0]+=S0[:,None]
        samples=[];start=0
        for grid,psf,ix in zip(regions,psfs,indices):
            block=spectrum[:,start:start+len(grid)].transpose(0,2,1);half=len(psf)//2
            padded=np.pad(block,((0,0),(0,0),(half,half)),mode='edge')
            out=np.stack([[correlate(row,psf/psf.sum(),mode='valid',method='direct') for row in pixel] for pixel in padded])
            samples.append(out[...,ix]);start+=len(grid)
        predicted=np.concatenate(samples,axis=-1)
        resid=(predicted-observed.reshape(-1,4,30)[begin:begin+len(p)])/sigma[None,:,None]
        squares+=np.sum(resid**2,axis=(0,2))
    result=np.sqrt(squares/(len(m)*30))
    if not np.isfinite(result).all(): raise ValueError('nonfinite observation metric')
    return result

def evaluate(prediction):
    config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
    thresholds={}
    for stokes in 'IQUV':
        key=stokes+'_whitened_rms';thresholds[key]={'comparison':'<=','boundary':finite_boundary(config,key+'_boundary')}
    with np.load(prediction,allow_pickle=False) as z:
        a=z['reconstruction']
        if a.shape!=(1,64,96,9) or a.dtype!=np.dtype('float64') or np.iscomplexobj(a) or not np.isfinite(a).all():
            raise ValueError('reconstruction must be finite real float64 (1,64,96,9)')
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:cavity=z['cavity_velocity'][0]
    native=a[0].copy();native[...,3]-=cavity+.45
    lower=np.array([0.,0.,0.,-13.,.001,.01,1e-5,0.,-10.])
    upper=np.array([6000.,np.pi,np.pi,13.,.2,120.,10.,10.,10.])
    domain_error=float(max(np.maximum(lower-native,0).max(),np.maximum(native-upper,0).max()))
    values={'parameter_domain_violation':domain_error}
    thresholds['parameter_domain_violation']={'comparison':'<=','boundary':1e-10}
    values.update(zip((c+'_whitened_rms' for c in 'IQUV'),map(float,observation_rms(a[0],HERE.parent/'data/raw_data.npz'))))
    passed=all(values[k]>=t['boundary'] if t['comparison']=='>=' else values[k]<=t['boundary'] for k,t in thresholds.items())
    return {'status':'pass' if passed else 'fail','passed':passed,'metrics':values,'thresholds':thresholds,'errors':[]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    try: report=evaluate(args.prediction);code=0 if report['passed'] else 1
    except Exception as exc: report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[str(exc)]};code=2
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc: print('report write failed: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__': sys.exit(main())
