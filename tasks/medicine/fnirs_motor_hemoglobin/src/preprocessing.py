"""Optical-density preprocessing extracted from MNE-Python (BSD-3-Clause).
Copyright the MNE-Python contributors; adapted 2026 for explicit array inputs.
"""
import numpy as np

SAMPLE_RATE_HZ = 8.928571428571429
WAVELENGTHS_NM = (760.0, 850.0)
PATHLENGTH_FACTOR = 6.0

def load_observations(path):
    """Load public NPZ, preserving float64 values; return independent arrays."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in ('intensity','distances','extinction','positions')}
    if arrays['intensity'].shape != (1,68,2,4860):
        raise ValueError('Expected complete recording (1,68,2,4860)')
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite observations/calibration')
    if np.any(arrays['intensity'] <= 0) or np.any(arrays['distances'] <= 0):
        raise ValueError('Intensity and distances must be positive')
    return arrays

def optical_density(intensity):
    """Return -ln(I/mean_time(I)), independently for each channel; dimensionless."""
    intensity = np.asarray(intensity, dtype=np.float64)
    if np.any(intensity <= 0) or not np.isfinite(intensity).all():
        raise ValueError('Optical density requires finite positive intensity')
    return -np.log(intensity / intensity.mean(axis=-1, keepdims=True))
