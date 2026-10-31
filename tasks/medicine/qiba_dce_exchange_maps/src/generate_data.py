"""Maintainer-only perturbations of existing author-simulated measurements.

These functions do not regenerate the bound phantom or its hidden parameters.
They add zero-mean Gaussian perturbations in observed signal units, with each
voxel's first-two-frame mean setting the relative standard deviation.  Neither
negative samples nor noisy outputs are clipped.  AIF and independently supplied
R10 calibration are copied unchanged.  The seed is experiment metadata, never
an input enabling reproduction of the hidden kinetic fields.
"""
import numpy as np

def perturb_signal(observations,relative_sigma,seed):
    if not np.isfinite(relative_sigma) or relative_sigma<0:
        raise ValueError('Relative perturbation sigma must be finite and nonnegative')
    result={k:np.array(v,copy=True) for k,v in observations.items()}
    signal=result['signal']
    scale=np.mean(signal[:,:2],axis=1,keepdims=True)*relative_sigma
    noise=np.random.RandomState(seed).normal(size=signal.shape)*scale
    result['signal']=signal+noise
    return result
