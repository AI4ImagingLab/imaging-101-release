"""Direct phase-contrast inverse and explicit source temporal gauge.

Copyright2025 Marta Maggioni; GPL-3.0-or-later. Adapted from original
Open4DFlow9e752d86644f52c789c3c9392886653feed33052, lines128-149.
No MRI library, BART reconstruction, hidden asset or upstream runtime import.
"""
import numpy as np
from .physics_model import phase_differences
from .preprocessing import native_order,VENC_M_S

def recover_velocity(encoding_images,venc_m_s=VENC_M_S,mask_fraction=.1,remove_temporal_mean=True):
    """Native x,y,z,time,4 -> relative velocity x,y,z,time,3 and diagnostics.

    Preserve source storage, principal angle, float32 arithmetic, mean-before-
    mask order and strict intensity threshold. Optional settings are labeled
    notebook sensitivity cases; defaults define the benchmark reconstruction.
    """
    if not np.isfinite(venc_m_s) or venc_m_s<=0:raise ValueError('VENC must be positive m/s')
    if not np.isfinite(mask_fraction) or not 0<mask_fraction<1:raise ValueError('mask_fraction must be in (0,1)')
    phases=phase_differences(encoding_images)
    velocity=(phases/np.pi)*venc_m_s
    means=np.empty(velocity.shape[:3]+(1,3),dtype=velocity.dtype)
    for axis in range(3):
        means[...,axis]=np.mean(velocity[...,axis],axis=3,keepdims=True)
        if remove_temporal_mean:velocity[...,axis]-=means[...,axis]
    magnitude=np.abs(encoding_images[...,0])
    mask=magnitude>mask_fraction*np.max(magnitude)
    velocity=velocity*mask[...,None]
    return velocity,dict(phase_differences=phases,temporal_velocity_mean=means,reference_magnitude=magnitude,mask=mask)

def reconstruct(images,mask_fraction=.1,remove_temporal_mean=True):
    """Public layout -> ({reconstruction:float32(1,27,3,54,67,100)},diagnostics)."""
    native=native_order(images)
    velocity,diagnostics=recover_velocity(native,mask_fraction=mask_fraction,remove_temporal_mean=remove_temporal_mean)
    out=np.ascontiguousarray(velocity.transpose(3,4,2,1,0))[None]
    return {'reconstruction':out},diagnostics
