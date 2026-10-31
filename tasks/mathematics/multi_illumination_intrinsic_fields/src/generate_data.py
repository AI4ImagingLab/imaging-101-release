"""Review-only synthetic perturbations of real author observations.

The bound benchmark is never generated here. Add independent Gaussian noise
in the original PNG/255 relative detector units, then clip only negative
intensity to zero. Sigma is an illustrative stress level, NOT a measured
camera noise parameter. The published reconstruction independently floors
intensity to3 before its logarithm. No hidden target or seed is used by the
public acquisition; fixed optional stress seed remains maintainer-only.
"""
import numpy as np
def perturb_observations(image,multiple,sigma,seed=20261022):
    if not np.isfinite(sigma) or sigma<0:raise ValueError('sigma must be nonnegative and finite')
    rng=np.random.default_rng(seed)
    return np.maximum(0,np.asarray(image)+rng.normal(0,sigma,np.shape(image))),np.maximum(0,np.asarray(multiple)+rng.normal(0,sigma,np.shape(multiple)))
