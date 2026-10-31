"""Explicit Stolt migration and closed-form diffusion/migration reconstruction.

Adapted from Stanford2020 MIT source; see LICENSE. No domain library/runtime.
"""
import numpy as np
import torch
from .preprocessing import preprocess
from .physics_model import frequency_mapping,paired_wiener,SCATTERING,THICKNESS

def migrate(measurement: torch.Tensor) -> torch.Tensor:
    """Return nonnegative float32(256,64,64) migrated squared wave envelope.

    The selected author implementation uses sqrt(max(input,0)), with no time
    multiplier. Positive-frequency sampling is linear only along time because
    both lateral frequency coordinates are integer. Invalid samples are zero.
    """
    if tuple(measurement.shape)!=(256,64,64):raise ValueError('Expected padded256x64x64')
    data=torch.sqrt(torch.clamp(measurement,min=0))
    transformed=torch.fft.fftshift(torch.fft.fftn(data))
    q=frequency_mapping()
    position=(q+1)*128
    invalid=(position<0)|(position>255)
    low=torch.clamp(torch.floor(position).long(),0,255)
    high=torch.clamp(low+1,0,255)
    fraction=torch.clamp(position,0,255)-low.float()
    # Paired arithmetic reproduces the original real/imag interpolation order.
    grid=torch.view_as_real(transformed)
    xx=torch.arange(64)[None,:,None].expand(256,64,64)
    yy=torch.arange(64)[None,None,:].expand(256,64,64)
    sampled=grid[low,xx,yy]*(1-fraction[...,None])+grid[high,xx,yy]*fraction[...,None]
    sampled*=~invalid[...,None]
    z=torch.arange(-128,128,dtype=torch.float32)[:,None,None]/128
    sampled*=torch.abs(z[...,None])/torch.clamp(q[...,None],min=1e-8)
    sampled*=(z>0)[...,None].float()
    wave=torch.fft.ifftn(torch.fft.ifftshift(torch.view_as_complex(sampled.contiguous())))
    # Keep separate squares, as in the source rather than abs() followed by square.
    return wave.real**2+wave.imag**2

def reconstruct(counts: np.ndarray,snr: float=2500.,mu_s: float=SCATTERING,thickness: float=THICKNESS) -> dict[str,np.ndarray]:
    """Compute all three complete comparisons on the bound raw observations.

    Return observations, migration_only, signed deconvolution_only and reconstruction
    as float32(128,32,32), with standard last15-range suppression only on combined.
    """
    b=preprocess(counts)
    padded=torch.nn.functional.pad(torch.from_numpy(b),(0,32,0,32,0,128))
    migration=migrate(padded).numpy()[:128,:32,:32].copy()
    deconvolved=paired_wiener(padded,snr,mu_s,thickness)
    result=migrate(deconvolved).numpy()[:128,:32,:32].copy()
    result[-15:]=0
    return dict(observations=b,migration_only=migration,deconvolution_only=deconvolved.numpy()[:128,:32,:32].copy(),reconstruction=result)
