"""Maintainer-only binomial-retention observations for sensitivity review.

The real bound count matrix is unchanged. The active notebook independently
retains each observed contact with probability0.5 or0.2, restarting original
RandomState(seed0) per case. Counts remain integer contact events, including
zeros; no hidden geometry or outcome-selected mask enters generation.
These are simulated perturbations of real observations, not new biological
replicates. Every originally supported locus remains in the comparison score.
"""
import numpy as np


def retain_contacts(counts,probability,seed=0):
    """Return integer matrix of independently retained contact events."""
    a=np.asarray(counts)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.issubdtype(a.dtype,np.integer) or np.any(a<0):raise ValueError('Nonnegative square integer contact counts required')
    if not np.isfinite(probability) or not 0<=probability<=1:raise ValueError('Retention probability must lie in[0,1]')
    return np.random.RandomState(seed).binomial(a,probability)
