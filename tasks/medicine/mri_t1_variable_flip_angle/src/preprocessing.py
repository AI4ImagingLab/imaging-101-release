"""Exact NIfTI-to-NPZ observation values; no spatial resampling."""
import numpy as np
def load_data(path):
    """Return float64 signal(128,128,2), B1(128,128), bool tissue mask."""
    with np.load(path,allow_pickle=False) as f:
        s=f['measurements'][0].copy();b=f['b1_map'][0].copy();m=f['tissue_mask'][0].astype(bool)
    if s.shape!=(128,128,2) or b.shape!=(128,128) or m.shape!=(128,128):raise ValueError('Unexpected observation shape')
    if s.dtype!=np.float64 or b.dtype!=np.float64:raise ValueError('Expected float64 observations')
    if np.any(b[m]<=0) or np.any(s[m]<=0):raise ValueError('Nonpositive masked measurement/calibration')
    return s,b,m
