"""Explicit elastic spectral stiffness/compliance for a bonded layer.

Adapted from pyTFM, Copyright Andreas Bauer et al., GPL-3.0-or-later.
"""
import numpy as np
from scipy.fft import fft2,ifft2

def stiffness(n,pitch,thickness,young,poisson):
    """Return Kxx,Kxy,Kyy (Pa/µm) on the even N×N Fourier grid.

    thickness=None is the infinite-half-space limit. DC is set to zero
    because image rigid translation carries no traction information.
    Cross terms at Nyquist lines are zero: this is the effective real-valued
    operator of upstream's inverse FFT, without non-Hermitian artefacts.
    """
    kx,ky=np.meshgrid(2*np.pi*np.fft.fftfreq(n,pitch),2*np.pi*np.fft.fftfreq(n,pitch))
    k=np.hypot(kx,ky);safe=k.copy();safe[0,0]=1
    nx=kx/safe;ny=ky/safe
    if thickness is None:
        transverse=young*k/(2*(1+poisson))
        longitudinal=young*k/(2*(1-poisson**2))
    else:
        r=k*thickness;safe_r=r.copy();safe_r[0,0]=1
        # Algebraic sech^2 form avoids large-r cosh overflow while preserving
        # the exact finite-layer Green tensor, not a half-space fallback.
        e=np.exp(-2*safe_r);sech2=4*e/(1+e)**2;tanh=(1-e)/(1+e)
        gamma=((3-4*poisson)+(1-2*poisson)**2*sech2+safe_r**2*sech2)/((3-4*poisson)*tanh+safe_r*sech2)
        transverse=young*k/(2*(1+poisson)*tanh)
        longitudinal=young*k*gamma/(2*(1-poisson**2))
    xx=transverse*ny**2+longitudinal*nx**2
    yy=transverse*nx**2+longitudinal*ny**2
    xy=(longitudinal-transverse)*nx*ny
    if n%2==0:
        corner=xy[n//2,n//2]
        xy[:,n//2]=0;xy[n//2,:]=0
        if thickness is not None:xy[n//2,n//2]=corner
    xx[0,0]=yy[0,0]=xy[0,0]=0
    return xx,xy,yy

def apply_spectral(vector,tensor):
    """Apply symmetric 2×2 spectral tensor to real vector (2,N,N)."""
    xx,xy,yy=tensor;u,v=fft2(vector,axes=(-2,-1))
    return ifft2(np.stack((xx*u+xy*v,xy*u+yy*v)),axes=(-2,-1)).real
