"""Explicit two-compartment filtration and MRI signal physics.
Extracted from dcmri (Apache-2.0); source revision and scope in validation.
"""
import numpy as np

def exponential_convolution(flux,transit,time):
    """Convolution with exp(-t/transit)/transit using linear input on each interval."""
    if transit==0:return np.asarray(flux).copy()
    f=np.asarray(flux);n=len(f);x=np.diff(time)/transit
    df=np.diff(f)/x;E=np.exp(-x);E0=1-E;E1=x-E0
    add=f[:-1]*E0+df*E1;g=np.zeros(n,dtype=E.dtype)
    for i in range(n-1):g[i+1]=E[i]*g[i]+add[i]
    return g

def compartment_concentration(arterial_plasma,time,parameters):
    """Return tissue-normalized plasma and tubular concentrations (M), shape (2,T)."""
    Fp,vp,FF,Tt=np.asarray(parameters,dtype=np.float64)
    Ft=FF*Fp;Tp=vp/(Fp+Ft)
    Cp=Tp*exponential_convolution(Fp*arterial_plasma,Tp,time)
    cp=Cp/vp
    Ct=Tt*exponential_convolution(Ft*cp,Tt,time)
    return np.stack((Cp,Ct))

def steady_signal(scale,R1,tr,flip_angle):
    """Spoiled steady-state GRE magnitude, preserving original operation order."""
    R1=np.asarray(R1);E=np.exp(-np.multiply(tr,R1));c=np.cos(flip_angle*np.pi/180)
    n=(1-E)/(1-c*E)
    with np.errstate(divide='ignore',invalid='ignore'):Mz=np.where(R1==0,n,n*R1/R1)
    return scale*np.sin(flip_angle*np.pi/180)*np.abs(Mz)

def arterial_concentration(signal,cal,sequence='SS'):
    """Invert arterial MRI calibration, preserving upstream out-of-range interpolation."""
    S=np.asarray(signal);R10=1/cal['blood_t1'];Sb=np.mean(S[:cal['n0']]);r1=cal['relaxivity']
    if sequence=='lin':return (R10*S/Sb-R10)/r1
    tr=cal['tr'];c=np.cos(cal['flip_angle']*np.pi/180);E0=np.exp(-tr*R10)
    Sn=S/Sb*(1-E0)/(1-c*E0)
    out=Sn>=1
    if np.any(out):
        indices=np.arange(len(Sn));Sn[out]=np.interp(indices[out],indices[~out],Sn[~out])
    ratio=(1-Sn)/(1-c*Sn)
    with np.errstate(divide='ignore',invalid='ignore'):
        R1=np.where(ratio==0,0,-np.log(ratio)/tr)
    return (R1-R10)/r1

def predict(parameters,time,arterial,cal,scale,sequence='SS',delay=0.):
    """Forward renal magnitude from calibrated arterial blood concentration."""
    ca=arterial if delay==0 else np.interp(time-delay,time,arterial,left=0)
    C=compartment_concentration(ca/(1-cal['H']),time,parameters)
    R1=1/cal['tissue_t1']+cal['relaxivity']*np.sum(C,axis=0)
    signal=scale*R1 if sequence=='lin' else steady_signal(scale,R1,cal['tr'],cal['flip_angle'])
    return signal,C,R1
