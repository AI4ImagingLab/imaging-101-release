"""New noise only for explicit sensitivity diagnostics, never bound observations.

The generator adds independent zero-mean Gaussian flux noise to all28x2048
sensor samples. Standard deviation is relative_sigma times the maximum absolute
observed input flux, in the same author unit U. Wavelengths and phases remain
unchanged. NumPy default_rng(PCG64) uses the explicit seed. No hidden truth or
target-derived calibration is read; these perturbations are not truth datasets.
"""
import numpy as np
def add_spectral_noise(raw,relative_sigma=.01,seed=123):
    if isinstance(relative_sigma,bool) or not isinstance(relative_sigma,(int,float)) or not np.isfinite(relative_sigma) or relative_sigma<0:raise ValueError('Invalid noise scale')
    copied={k:np.array(v,copy=True) for k,v in raw.items()};sigma=relative_sigma*np.max(np.abs(copied['spectrum']))
    copied['spectrum']+=np.random.default_rng(seed).normal(0,sigma,size=copied['spectrum'].shape)
    return copied
