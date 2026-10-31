"""MIT extraction from Logan Bishop-Van Horn, copyright2024.
Source revision443b3a1c8efa3f9a37c1be5ce01e2e2da84d0aa0; see LICENSE/NOTICE.
Odd-grid bin/origin and explicit DC fixes are documented in validation.
"""
import numpy as np
from scipy.constants import mu_0 as MU0

STANDOFF=1.

def frequency_grid(shape,dx,dy):
    kx,ky=np.meshgrid(2*np.pi*np.fft.fftshift(np.fft.fftfreq(shape[1],d=dx)),2*np.pi*np.fft.fftshift(np.fft.fftfreq(shape[0],d=dy)))
    return kx,ky,np.sqrt(kx**2+ky**2)

def psf_transfer(psf):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))

def hanning(k,cutoff=1.):
    maximum=np.sqrt(cutoff**2+cutoff**2)
    h=.5*(1+np.cos(np.pi*k/maximum));h[k>maximum]=0
    return h

def invert_spectrum(flux_k,psf_k,kx,ky,k,h,standoff=STANDOFF):
    factor=np.zeros_like(flux_k);nonzero=k>0
    factor[nonzero]=2j*flux_k[nonzero]*h[nonzero]/(MU0*np.exp(-k[nonzero]*standoff)*k[nonzero]*psf_k[nonzero])
    return np.stack([-ky*factor,kx*factor]),factor

def forward_spectrum(j_k,psf_k,kx,ky,k,standoff=STANDOFF):
    """Effective author signed flux convention; not calibrated absolute +Bz."""
    flux=np.zeros_like(psf_k);nonzero=k>0
    flux[nonzero]=MU0/(2j)*np.exp(-k[nonzero]*standoff)*psf_k[nonzero]*(kx[nonzero]*j_k[1,nonzero]-ky[nonzero]*j_k[0,nonzero])/k[nonzero]
    return flux

def diagnostics(current,data,cutoff=1.):
    from .preprocessing import pad_flux
    flux,x,y=pad_flux(data);kx,ky,k=frequency_grid(flux.shape,x[1]-x[0],y[1]-y[0]);p=psf_transfer(data['psf_coarse'])
    jk=np.fft.fftshift(np.fft.fft2(current,axes=(-2,-1)),axes=(-2,-1))
    target=np.fft.fftshift(np.fft.fft2(flux))*hanning(k,cutoff);target[k==0]=0
    predicted=forward_spectrum(jk,p,kx,ky,k)
    div_denominator=np.linalg.norm(k*np.sqrt(abs(jk[0])**2+abs(jk[1])**2))
    return {'filtered_forward_residual':float(np.linalg.norm(predicted-target)/np.linalg.norm(target)),
            'divergence_ratio':float(np.linalg.norm(kx*jk[0]+ky*jk[1])/div_denominator) if div_denominator else 0.,
            'dc_ratio':float(np.linalg.norm(np.mean(current,axis=(-2,-1)))/max(np.sqrt(np.mean(current**2)),np.finfo(float).tiny)),
            'prediction_k':predicted,'target_k':target}
