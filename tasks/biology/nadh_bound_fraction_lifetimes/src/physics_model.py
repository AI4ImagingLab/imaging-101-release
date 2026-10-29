"""Explicit periodic fluorescence Fourier moments and independent-standard calibration."""
import numpy as np
FREQUENCY=80.
REFERENCE_LIFETIME=3.05
OMEGA=2*np.pi*FREQUENCY*1e-3
def calibrated_phasors(signal,reference):
    total=signal.sum(axis=-1,dtype=np.float64);mean=total/signal.shape[-1]
    ft=np.fft.rfft(signal,axis=-1);indices=np.array([1,2]);raw=np.moveaxis(ft[...,indices].conj(),-1,0)
    raw=np.divide(raw,total[None],out=np.full_like(raw,np.nan),where=total[None]>0)
    pooled=np.sum(reference,axis=(0,1),dtype=np.float64);ref=np.fft.rfft(pooled)[indices].conj()/pooled.sum()
    theory=1/(1-1j*OMEGA*indices*REFERENCE_LIFETIME)
    return mean,raw*(theory/ref)[:,None,None]
def mixture_phasors(lifetimes,fractions):
    t=np.asarray(lifetimes);f=np.asarray(fractions);omega=OMEGA*np.arange(1,3)
    return np.sum(f[None]/(1-1j*omega.reshape((2,)+(1,)*t.ndim)*t[None]),axis=1)
