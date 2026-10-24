"""Diagnostic synthetic relaxivity perturbations, never replacement benchmark data.

Adds independently sampled zero-mean Gaussian relaxation-rate noise (s^-1) to
all supported concentration curves. Sigma is relative to the complete observed
supported dynamic range; NumPy PCG64 seed is explicit. These are mathematical
noise-sensitivity experiments, not measured data or independent patient truth.
The bound raw_data.npz always remains the real original GRE observations.
"""
import numpy as np

def perturb_concentration(concentration,relative_sigma=.01,seed=124):
    """Return newly generated N-by-time concentration observations in s^-1."""
    if not np.isfinite(relative_sigma) or relative_sigma<0:raise ValueError('Invalid noise scale')
    y=np.asarray(concentration,dtype=np.float64)
    return y+np.random.default_rng(seed).normal(0,relative_sigma*np.ptp(y),size=y.shape)
