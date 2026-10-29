"""Explicit complex phase-contrast acquisition model; GPL-3.0-or-later.

Selected source: Copyright2025 Marta Maggioni, Open4DFlow. Reference-encoded
complex products and principal angles are computed locally using NumPy only.
"""
import numpy as np
from .preprocessing import VENC_M_S

def encode_phasors(velocity,venc_m_s=VENC_M_S):
    """Velocity (...,3), m/s -> reference plus three unit complex encodings.

    Positive acquisition-axis velocity advances phase by pi*v/VENC. This
    pure analytic helper creates only independent test cases, not bound data.
    """
    velocity=np.asarray(velocity)
    if velocity.shape[-1]!=3:raise ValueError('Expected3velocity directions')
    ref=np.ones(velocity.shape[:-1]+(1,),dtype=np.complex128)
    return np.concatenate([ref,np.exp(1j*np.pi*velocity/venc_m_s)],axis=-1)

def phase_differences(encoding_images):
    """Native x,y,z,time,4 complex images -> x,y,z,time,3 radians in [-pi,pi]."""
    if encoding_images.ndim!=5 or encoding_images.shape[-1]!=4:raise ValueError('Expected native5D four-encoding image')
    return np.stack([np.angle(encoding_images[...,d]*np.conj(encoding_images[...,0])) for d in (1,2,3)],axis=-1)
