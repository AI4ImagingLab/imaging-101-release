"""Generate maintainer-only sensitivity observations, never the measured primary.

The benchmark raw_data.npz is a lossless conversion of existing real HDR frames.
This helper only adds diagnostic Gaussian perturbations for the active review
Notebook. It writes no file and does not define a hidden target or primary seed.
The frozen study uses per-frame measured spatial RMS, fractions.005/.01/.02 and
diagnostic seed31917. Native lower clipping occurs later in reconstruct, exactly
as before this helper extraction; its newly clipped fraction remains measured.
"""
import numpy as np

def add_gaussian_perturbation(intensity,frame_rms,fraction,seed=31917):
    """Return full-grid float64 noisy observations, retaining native RNG order."""
    intensity=np.asarray(intensity);frame_rms=np.asarray(frame_rms)
    if intensity.ndim!=3 or frame_rms.shape!=(len(intensity),):raise ValueError('Frame images and matching RMS vector required')
    if not np.isfinite(intensity).all() or not np.isfinite(frame_rms).all() or np.any(frame_rms<0):raise ValueError('Finite observations and nonnegative RMS required')
    if not np.isfinite(fraction) or fraction<0:raise ValueError('Nonnegative finite noise fraction required')
    generator=np.random.default_rng(seed);noisy=intensity.astype(np.float64)
    for i in range(len(intensity)):
        noisy[i]+=generator.normal(0,fraction*frame_rms[i],size=intensity.shape[1:])
    return noisy
