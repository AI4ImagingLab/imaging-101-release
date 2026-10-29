"""Flat-sky spin-2 shear/convergence relations, with the DC mass sheet fixed to zero.
Adapted from Lenspack (CosmoStat), MIT; see NOTICE.
"""
import numpy as np
def kernels(shape):
    """Return dimensionless P1,P2 on NumPy's FFT grid; axes are (north,east)."""
    fy,fx=np.meshgrid(np.fft.fftfreq(shape[0]),np.fft.fftfreq(shape[1]),indexing="ij")
    q=fx*fx+fy*fy;safe=np.where(q==0,1,q)
    return (fx*fx-fy*fy)/safe,2*fx*fy/safe
def forward(e,b):
    """Return real shear maps gamma1,gamma2 from dimensionless E/B convergence."""
    p1,p2=kernels(e.shape);fe,fb=np.fft.fft2(e),np.fft.fft2(b)
    return np.fft.ifft2(p1*fe-p2*fb).real,np.fft.ifft2(p2*fe+p1*fb).real
def inverse(g1,g2):
    """KS93 inversion with zero mean, periodic boundary and unnormalised forward FFT."""
    p1,p2=kernels(g1.shape);f1,f2=np.fft.fft2(g1),np.fft.fft2(g2)
    return np.fft.ifft2(p1*f1+p2*f2).real,np.fft.ifft2(-p2*f1+p1*f2).real
