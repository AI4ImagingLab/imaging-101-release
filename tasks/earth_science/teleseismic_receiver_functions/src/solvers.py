"""Explicit water-level, Toeplitz and iterative receiver-function inverses.
Adapted from rf, Copyright 2013-2024 Tom Eulenfeld, Hannah Mark; MIT LICENSE.
"""
import numpy as np
from scipy.fftpack import fft, ifft, next_fast_len
from scipy.signal import correlate
from scipy.linalg import solve_toeplitz
from .physics_model import filters
from .preprocessing import SAMPLING_RATE
def waterlevel(zrt, waterlevel=.05, gauss=.5, shift=30.):
    """Return normalized RF (3,N) and diagnostic dictionary. Source is Z.
    H=IFFT(G exp(-i*w*t0) D conj(S)/max(|S|²,alpha max|S|²)).
    Normalize all components by the maximum real Z self-response.
    """
    zrt=np.asarray(zrt); n=zrt.shape[-1]; nf=next_fast_len(n)
    filt,f=filters(nf,gauss,shift); spec=fft(zrt,nf,axis=-1); s=spec[0]
    power=np.abs(s*np.conj(s)); clipped=np.maximum(power,waterlevel*power.max())
    rf=ifft(filt*spec*np.conj(s)/clipped,axis=-1)[...,:n]
    norm=1/np.max(rf[0])
    return (rf*norm).real,dict(norm=float(norm.real),power=clipped,source_conj=np.conj(s),filter=filt,freq=f)
def _cross(a,b,n,shift=0):
    if shift>0:a=np.pad(a,(2*abs(shift),0))
    elif shift<0:a=np.pad(a,(0,2*abs(shift)))
    dif=len(a)-len(b)+1-n
    if dif>0:b=np.pad(b,((dif+1)//2,dif//2))
    else:a=np.pad(a,((-dif+1)//2,(-dif)//2))
    return correlate(a,b,'valid')
def time_domain(zrt,spiking=.1,shift=30.):
    """Source autocorrelation Toeplitz inverse with diagonal spiking.
    Matches rf's source-energy scaling, absolute maximum normalization.
    """
    n=zrt.shape[-1]; s=zrt[0]; ac=correlate(np.pad(s,(0,n-1)),s,'valid')
    ac=ac/ac[0]; ac[0]+=spiking
    out=np.array([solve_toeplitz(ac,_cross(r,s,n,int(shift*SAMPLING_RATE-n//2))) for r in zrt])
    return out/np.max(np.abs(out[0]))
def iterative(zrt, gauss=.5, shift=30., itmax=400, minderr=.001):
    """Ligorria-Ammon matching-pursuit spikes with Gaussian filtered residual.
    minderr is percentage-point squared-error improvement, not relative norm.
    Negative spikes allowed; causal lag mute; maximum 400 iterations.
    """
    n=zrt.shape[-1]; nf=next_fast_len(2*n); dt=1/SAMPLING_RATE
    f=np.fft.fftfreq(nf,dt); gf=np.exp(-.5*(f/gauss)**2)
    ss=fft(zrt[0],nf)*gf; ps=np.sum(ifft(ss).real**2)
    results=[]; iterations=[]; histories=[]
    for r in zrt:
        target=ifft(fft(r,nf)*gf).real; pr=np.sum(target**2)
        remaining=target.copy(); spikes=np.zeros(nf); old=1.; delta=100*pr+minderr
        hist=[]
        while abs(delta)>minderr and len(hist)<itmax:
            corr=ifft(fft(remaining)*np.conj(ss)).real/ps/dt
            corr[n:nf-int(shift*SAMPLING_RATE)]=0.; index=np.argmax(np.abs(corr)); spikes[index]+=corr[index]
            estimate=ifft(fft(spikes)*ss).real*dt; remaining=target-estimate
            error=np.sum(remaining**2)/pr; hist.append(error); delta=100*(old-error); old=error
        results.append(ifft(fft(spikes)*gf*np.exp(-2j*np.pi*f*shift)).real[:n])
        iterations.append(len(hist)); histories.append(hist)
    out=np.array(results); out/=np.max(np.abs(out[0]))
    return out,dict(iterations=iterations,histories=histories)
