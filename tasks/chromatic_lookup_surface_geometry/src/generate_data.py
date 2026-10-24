"""Review-only synthetic normalized-color perturbations of real observations.

The benchmark remains the actual measured uint16 pattern and white exposures.
For noise sensitivity, add independent zero-mean Gaussian noise with specified
standard deviation in dimensionless normalized RGB units to supported entries.
Keep negative draws, zero unsupported entries, and do not modify source arrays.
This is a controlled model perturbation, not a calibrated camera noise claim.
"""
import numpy as np

def perturb_normalized(normalized,mask,sigma,seed=232323):
    q=np.asarray(normalized,dtype=np.float32);m=np.asarray(mask,dtype=bool)
    if q.shape[:-1]!=m.shape or q.shape[-1]!=3 or not np.isfinite(q).all() or not np.isfinite(sigma) or sigma<0:raise ValueError('Finite RGB, matching support and nonnegative noise required')
    noise=np.random.default_rng(seed).normal(0,sigma,q.shape).astype(np.float32)
    result=q.copy();result[m]+=noise[m];result[~m]=0
    return result
