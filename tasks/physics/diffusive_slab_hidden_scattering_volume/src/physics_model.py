"""Explicit finite-slab diffusion and the selected author spectral conventions.

Adapted mathematical expressions: Copyright2020Stanford University, MIT; see LICENSE.
No upstream module is imported. All arrays use(time,x,y), zero-padded to256x64x64.
"""
import numpy as np
import torch

C0=3e8
REFRACTIVE_INDEX=1.12
SCATTERING=262.
ABSORPTION=.53
THICKNESS=.0254
EXTRAPOLATION=.0036
SCAN_WIDTH=.7
PATH_RANGE=4.

def diffusion_kernel(mu_s: float=SCATTERING,thickness: float=THICKNESS) -> np.ndarray:
    """Normalized double-pass circular impulse response, float64(256,64,64).

    Preserve author inclusive spatial/time lattices and41 image pairs. At t0 the
    original uses a1-second placeholder which underflows to zero; keep it explicit.
    """
    if not np.isfinite(mu_s) or mu_s<=0 or not np.isfinite(thickness) or thickness<=0:raise ValueError('Invalid slab')
    velocity=C0/REFRACTIVE_INDEX
    t=np.linspace(0,PATH_RANGE,256)[:,None,None]/velocity
    t[0]=1.
    lateral=np.linspace(-SCAN_WIDTH,SCAN_WIDTH,64)
    radius=np.sqrt(lateral[None,:,None]**2+lateral[None,None,:]**2)
    D=1/(3*(ABSORPTION+mu_s)); z0=1/mu_s
    images=np.arange(-20,21)[None,None,:]
    z1=thickness*(1-2*images)-4*images*EXTRAPOLATION-z0
    z2=thickness*(1-2*images)-(4*images-2)*EXTRAPOLATION+z0
    term=(z1*np.exp(-z1**2/(4*D*velocity*t))-z2*np.exp(-z2**2/(4*D*velocity*t))).sum(axis=-1)[...,None]
    k=(4*np.pi*D*velocity)**(-1.5)*t**(-2.5)*np.exp(-ABSORPTION*velocity*t-radius**2/(4*D*velocity*t))*term
    k/=k.sum()
    k=np.roll(np.roll(k,-32,axis=1),-32,axis=2)
    H=np.fft.fftn(k)
    return np.abs(np.fft.ifftn(H*H))

def transfer_function(mu_s: float=SCATTERING,thickness: float=THICKNESS) -> torch.Tensor:
    """Return float32 real/imag transfer(256,64,64,2), CPU."""
    k=torch.from_numpy(diffusion_kernel(mu_s,thickness).astype(np.float32))
    return torch.view_as_real(torch.fft.fftn(k))

def multiply_pair(a: torch.Tensor,b: torch.Tensor) -> torch.Tensor:
    """Explicit complex product with last-axis real/imag representation."""
    return torch.stack((a[...,0]*b[...,0]-a[...,1]*b[...,1],a[...,0]*b[...,1]+a[...,1]*b[...,0]),dim=-1)

def convolve(x: torch.Tensor,adjoint: bool=False) -> torch.Tensor:
    """Double-pass circular diffusion, or its exact discrete transpose."""
    h=torch.view_as_complex(transfer_function().contiguous())
    if adjoint:h=h.conj()
    return torch.fft.ifftn(torch.fft.fftn(x)*h).real

def paired_wiener(x: torch.Tensor,snr: float=2500.,mu_s: float=SCATTERING,thickness: float=THICKNESS) -> torch.Tensor:
    """Real deconvolution with original paired-reciprocal arithmetic, float32.

    Its frequency response equals conventional Wiener plus i*conj(H)*snr.
    For Hermitian H, the additional inverse-transform term is pure imaginary
    in exact arithmetic. Preserve source float32 cancellation and discard the
    imaginary output explicitly; do not pretend the paired reciprocal is the
    conventional complex reciprocal. Independent backend bounds cover roundoff.
    """
    if not np.isfinite(snr) or snr<=0:raise ValueError('Positive finite SNR required')
    h=transfer_function(mu_s,thickness)
    hc=h.clone(); hc[...,1]*=-1
    power=multiply_pair(h,hc)
    inv=multiply_pair(hc,1/(power+1/snr))
    product=multiply_pair(torch.view_as_real(torch.fft.fftn(x)),inv)
    return torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(product.contiguous())))[...,0]

def frequency_mapping() -> torch.Tensor:
    """Normalized positive temporal Stolt coordinate for every output frequency."""
    z,x,y=torch.meshgrid(torch.arange(-128,128),torch.arange(-32,32),torch.arange(-32,32),indexing='ij')
    z=z.float()/128; x=x.float()/32; y=y.float()/32
    # Float32 scalar arithmetic matches the independently frozen upstream grid.
    width=torch.tensor([SCAN_WIDTH],dtype=torch.float32); mrange=torch.tensor([PATH_RANGE],dtype=torch.float32)
    return torch.sqrt(torch.abs(((32*mrange)/(128*width*4))**2*(x*x+y*y)+z*z))
