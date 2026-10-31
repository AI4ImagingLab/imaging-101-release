"""Explicit filtered inverse; MIT adaptation, source attribution in NOTICE."""
import numpy as np
from .preprocessing import pad_flux
from .physics_model import frequency_grid,psf_transfer,hanning,invert_spectrum

def reconstruct(data,cutoff=1.,standoff=1.,pad_mode='linear_ramp',use_psf=True):
    flux,x,y=pad_flux(data,pad_mode);dx=x[1]-x[0];dy=y[1]-y[0]
    fk=np.fft.fftshift(np.fft.fft2(flux));pk=psf_transfer(data['psf_coarse']) if use_psf else np.ones(flux.shape,dtype='complex128')
    kx,ky,k=frequency_grid(flux.shape,dx,dy);h=hanning(k,cutoff)
    jk,factor=invert_spectrum(fk,pk,kx,ky,k,h,standoff)
    full=np.fft.ifft2(np.fft.ifftshift(jk,axes=(-2,-1)),axes=(-2,-1))
    imaginary=max(np.linalg.norm(z.imag)/np.linalg.norm(z.real) for z in full)
    if imaginary>1e-10:raise ValueError('Non-Hermitian inverse field: imaginary part cannot be silently discarded')
    return {'reconstruction':full.real[None].astype('float64')},{'padded_mag':flux,'padded_xs':x,'padded_ys':y,'mag_k':fk,'psf_k':pk,'Kx':kx,'Ky':ky,'K':k,'H':h,'factor':factor,'jx_k':jk[0],'jy_k':jk[1],'full_complex':full,'imaginary_ratio':float(imaginary)}
