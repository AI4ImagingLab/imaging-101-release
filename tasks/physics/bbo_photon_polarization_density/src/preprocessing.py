"""Strict loading of corrected photon counts and known analyzer projectors."""
from pathlib import Path
import numpy as np
def load_data(path="data/raw_data.npz"):
    """Return counts (36,) float64 and projectors (36,4,4) complex128."""
    with np.load(Path(path), allow_pickle=False) as z:
        counts=z['counts'][0].copy(); projectors=z['projectors'][0].copy()
    if counts.shape!=(36,) or projectors.shape!=(36,4,4): raise ValueError('Invalid acquisition shape')
    if not np.isfinite(counts).all() or not np.isfinite(projectors).all() or np.any(counts<0): raise ValueError('Invalid acquisition values')
    return counts,projectors
