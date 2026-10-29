"""Explicit apparent saturation-component model; not exchange-rate inference."""
import numpy as np

POOL_NAMES=('Water','MT','Amide','Creatine','NOE -3.5','NOE -1.6')

def lorentzian(x,amplitude,width,center):
    num=amplitude*.25*width**2
    den=.25*width**2+(x-center)**2
    return num/den

def background(x,*p):
    return 1-lorentzian(x,*p[:3])-lorentzian(x,*p[3:6])

def pool_sum(x,*p):
    out=np.zeros_like(x)
    for k in range(0,len(p),3):out+=lorentzian(x,*p[k:k+3])
    return out

def predict(offsets,parameters,field_shift):
    """(...,6,3) parameters -> (...,N) unit-normalized Z spectra."""
    x=np.asarray(offsets)-np.asarray(field_shift)[...,None]
    out=np.ones_like(x)
    for k in range(6):
        out-=lorentzian(x,parameters[...,k,0,None],parameters[...,k,1,None],parameters[...,k,2,None])
    return out

def lorentzian_jacobian(x,amplitude,width,center):
    q=.25*width**2;delta=x-center;den=q+delta**2
    return np.stack([q/den,amplitude*.5*width*delta**2/den**2,2*amplitude*q*delta/den**2],axis=-1)
