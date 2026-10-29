"""Discrete periodic angular spectrum propagation, using upstream float32 grids.

Adapted from Fringe, Copyright 2020 Raindoss; MIT license in task LICENSE.
All distances are in micrometres. FFTs use NumPy's unscaled forward convention.
"""
import numpy as np
from .preprocessing import PIXEL_SIZE,WAVELENGTH
K=2*np.pi/WAVELENGTH
def wave_grid(shape,pixel_size=PIXEL_SIZE):
    if len(shape)!=2 or any(int(n)!=n or n<=0 or n%2 for n in shape):
        raise ValueError("wave_grid supports positive even two-dimensional shapes only")
    axes=[]
    for n in shape:
        u=1/(float(pixel_size)*n)
        low=-(n/2)*u if n%2==0 else -((n-1)/2)*u
        high=(n/2-1)*u if n%2==0 else ((n-1)/2)*u
        axes.append(np.fft.fftshift(np.linspace(low*2*np.pi,high*2*np.pi,n)))
    grid=np.array(np.meshgrid(*axes,indexing='ij'),dtype='float32')
    return np.sum(grid*grid,axis=0)
def transfer(shape,distance,pixel_size=PIXEL_SIZE,wavenumber=K):
    kt2=wave_grid(shape,pixel_size)
    delta=wavenumber*wavenumber-kt2
    root=np.sqrt(np.abs(delta))
    return np.exp(np.where(delta>=0,1j*root*distance,-root*abs(distance)+0j))
def propagate(field,distance,pixel_size=PIXEL_SIZE,wavenumber=K):
    return np.fft.ifft2(transfer(field.shape,distance,pixel_size,wavenumber)*np.fft.fft2(field))
def predicted_amplitudes(field,distances):
    return np.stack([abs(propagate(field,np.float64(d))) for d in distances])
def amplitude_residual(field,amplitudes,distances):
    p=predicted_amplitudes(field,distances)
    return float(np.linalg.norm(p-abs(amplitudes))/np.linalg.norm(abs(amplitudes).astype(np.float64)))
