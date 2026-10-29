"""Maintainer robustness diagnostics; never used to create the benchmark input.

The bound input is the unchanged existing real author mesh. This helper creates
explicitly labeled synthetic perturbations of those observed coordinates for
notebook sensitivity experiments: independent zero-mean Gaussian displacement
of each coordinate, with standard deviation in author mesh coordinate units.
Topology, region labels and force references are never used to generate noise.
These perturbations are not a calibrated segmentation noise distribution or
new author observations; they are not saved in data/raw_data.npz.
"""
import numpy as np

def perturb_geometry(observations,sigma,seed):
    """Copy observed geometry and add documented Gaussian coordinate jitter."""
    if sigma<0:raise ValueError('sigma must be nonnegative')
    result={k:np.array(v,copy=True) for k,v in observations.items()}
    result['vertices']=result['vertices']+np.random.default_rng(seed).normal(0,sigma,result['vertices'].shape)
    return result
