"""Maintainer-only sensitivity observations; never creates the primary real data.

Add independent zero-mean Gaussian noise to already normalized measured spectra.
sigma is dimensionless normalized fluorescence; seed affects this review only.
No hidden material object, magnetic target or acquisition calibration is used.
"""
import numpy as np
def perturb_spectra(data,sigma,seed):
    return np.asarray(data)+np.random.default_rng(seed).normal(0,sigma,np.shape(data))

