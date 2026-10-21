"""Spatial lock-in and local least-squares phase-to-displacement projection."""
import numpy as np
from scipy.ndimage import fourier_gaussian

def wrap_phase(a): return (a+np.pi)%(2*np.pi)-np.pi

def lockin(image,k,sigma):
    y,x=np.ogrid[:image.shape[0],:image.shape[1]]
    return np.fft.ifft2(fourier_gaussian(np.fft.fft2(image*np.exp(2j*np.pi*(y*k[0]+x*k[1]))),sigma=sigma))

def adaptive_lockin(image,k,sigma,kw,kstep):
    y,x=np.ogrid[:image.shape[0],:image.shape[1]]
    best=np.zeros(image.shape,complex);selected=np.zeros((2,)+image.shape)
    for ky in np.arange(k[0]-kw,k[0]+kw,kstep):
        for kx in np.arange(k[1]-kw,k[1]+kw,kstep):
            v=lockin(image,(ky,kx),sigma)*np.exp(-2j*np.pi*((ky-k[0])*y+(kx-k[1])*x))
            mask=np.abs(v)>np.abs(best);best[mask]=v[mask];selected[:,mask]=np.array([ky,kx])[:,None]
    return best,selected

def weighted_projection(phase_differences,kvecs,weights):
    # Same leading-pixel weights as upstream: differences do not average edge weights.
    b=phase_differences;w=weights[:,:b.shape[1],:b.shape[2]]
    K=2*np.pi*kvecs;w2=w*w
    aa=np.sum(w2*K[:,0,None,None]**2,axis=0);bb=np.sum(w2*K[:,0,None,None]*K[:,1,None,None],axis=0);cc=np.sum(w2*K[:,1,None,None]**2,axis=0)
    r0=np.sum(w2*K[:,0,None,None]*b,axis=0);r1=np.sum(w2*K[:,1,None,None]*b,axis=0);det=aa*cc-bb*bb
    if np.any(det<=0):raise ValueError('Reference vectors or local weights are rank deficient')
    return np.stack([(cc*r0-bb*r1)/det,(aa*r1-bb*r0)/det])
