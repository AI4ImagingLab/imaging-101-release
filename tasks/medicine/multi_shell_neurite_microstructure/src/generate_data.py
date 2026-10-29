"""Maintainer-only sensitivity observations; no new bound acquisition.

The active notebook adds independent Gaussian intensity noise of standard
deviation relative_sigma times the measured voxel b0, to every positive-b
directional measurement. The measured b0 and fixed support are unchanged.
RandomState(seed20261022) is restarted for each1%/3% sensitivity case, exactly
preserving the original notebook experiment. Intensities remain scanner units;
negative perturbed signals are retained, without clipping or new masking.
These generated observations are numerical diagnostics, not real replicates
or a scanner noise calibration. Standard reconstruction uses only real input.
"""
import numpy as np


def add_gaussian_noise(observed,bvalues,relative_sigma,seed=20261022):
    """Return float64(n,acquisitions) perturbation, preserving measured b0."""
    y=np.asarray(observed,dtype=float);b=np.asarray(bvalues).ravel()
    if y.ndim!=2 or y.shape[1]!=len(b) or not np.isfinite(y).all() or not np.isfinite(b).all() or not np.any(b==0):raise ValueError('Finite signals, matching b values and observed b0 required')
    if not np.isfinite(relative_sigma) or relative_sigma<0:raise ValueError('Nonnegative finite relative noise required')
    s0=y[:,b==0].mean(1);rng=np.random.RandomState(seed);perturbed=y.copy()
    perturbed[:,b>0]+=rng.normal(size=(len(y),int((b>0).sum())))*(relative_sigma*s0[:,None])
    return perturbed
