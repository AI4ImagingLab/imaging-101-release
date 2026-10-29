"""Explicit unweighted multi-distance alternating amplitude projections.

Adapted from Fringe, Copyright 2020 Raindoss; MIT license in task LICENSE.
The optimization field resides at the first sensor plane between sweeps.
"""
import numpy as np
from .preprocessing import DISTANCES
from .physics_model import transfer,amplitude_residual
ITERATIONS=100
def reconstruct(amplitudes,iterations=ITERATIONS,distances=DISTANCES,record_every=5):
    a=np.asarray(amplitudes)
    if a.ndim!=3 or len(a)!=len(distances) or len(a)<2:
        raise ValueError('at least two amplitude planes with matching distances required')
    if not isinstance(iterations,int) or iterations<0:raise ValueError('nonnegative integer iterations required')
    if not np.isfinite(a).all():raise ValueError('finite amplitudes required')
    z=-np.asarray(distances,dtype=float)
    h=a[0].copy();history=[]
    kernels={float(d):transfer(h.shape,np.float64(d)) for d in np.unique(np.diff(z).tolist()+(-np.diff(z)).tolist()+[z[0]])}
    def step(v,d):return np.fft.ifft2(kernels[float(d)]*np.fft.fft2(v))
    for i in range(iterations):
        for j in range(len(a)-1):
            rec=step(h,z[j]-z[j+1]); phase=np.exp(np.zeros_like(rec,dtype='float32')+1j*np.angle(rec));h=a[j+1]*phase
        last=len(a)-1
        for j in range(len(a)-1):
            rec=step(h,z[last-j]-z[last-j-1]);phase=np.exp(np.zeros_like(rec,dtype='float32')+1j*np.angle(rec));h=a[last-j-1]*phase
        if record_every and ((i+1)%record_every==0 or i+1==iterations):
            obj=step(h,z[0]);history.append((i+1,amplitude_residual(obj,a,distances)))
    return step(h,z[0]),np.asarray(history,dtype=float).reshape(-1,2)
