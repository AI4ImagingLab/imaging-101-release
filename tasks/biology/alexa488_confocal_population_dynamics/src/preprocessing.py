from pathlib import Path
import numpy as np

def load_observations(path):
    """Load batch-first float64 lag times (ms) and dimensionless correlation."""
    with np.load(Path(path), allow_pickle=False) as a:
        t, g = a['lag_ms'][0].copy(), a['correlation'][0].copy()
    if t.shape != (200,) or g.shape != t.shape or not np.all(np.isfinite([t,g])) or np.any(np.diff(t)<=0):
        raise ValueError('Expected 200 finite ordered lag times and correlation values')
    return t,g
