"""Strict bound detector-event input; no synthetic observations or target data."""
from pathlib import Path
import numpy as np

def load_data(path):
    """Read float32(1,522963,6,3) detector mm and return unbatched events."""
    with np.load(Path(path), allow_pickle=False) as z:
        h = z['detector_hits']
    if h.shape != (1, 522963, 6, 3) or h.dtype != np.float32:
        raise ValueError('Expected float32(1,522963,6,3) detector hits')
    if not np.isfinite(h).all():
        raise ValueError('Nonfinite detector hit')
    return h[0]
