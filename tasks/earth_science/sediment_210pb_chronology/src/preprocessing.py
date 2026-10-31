"""Measured profiles and algorithmic normal proposals; no hidden reference access."""
from pathlib import Path
import random
import numpy as np
from scipy.special import ndtri

def load_observations(path=Path('data/raw_data.npz')):
    with np.load(path,allow_pickle=False) as data:
        mass,activity,error=(data[k][0].copy() for k in ['mass_depth','activity','activity_error'])
    if mass.shape!=(14,) or not np.all(np.isfinite([mass,activity,error])):raise ValueError('Fourteen finite observed layers required')
    if np.any(np.diff(np.r_[0.,mass])<=0) or np.any(error<=0):raise ValueError('Positive thicknesses and errors required')
    return mass,activity,error

def canonical_proposals(n=14,seed=20261029):
    """Author canonical quantiles, independently shuffled and rounded to four decimals.

    Reproduce the library generator's preceding sizes 5..n. This seed is only
    reproducible algorithm state, never an acquisition parameter or hidden truth.
    """
    rng=random.Random(seed)
    for size in range(5,n+1):
        quantiles=ndtri((np.arange(size)+.5)/size)
        first=quantiles.copy();second=quantiles.copy()
        rng.shuffle(first);rng.shuffle(second)
    return np.round(first,4),np.round(second,4)
