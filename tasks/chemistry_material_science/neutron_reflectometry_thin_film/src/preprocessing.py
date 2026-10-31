from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load_observations(path=ROOT/'data/raw_data.npz'):
    """Return q, R, standard error, resolution FWHM as float64 (N,) arrays."""
    with np.load(path,allow_pickle=False) as d:
        a=tuple(d[k][0].copy() for k in ('q','reflectivity','standard_error','q_fwhm'))
    if any(x.ndim!=1 or x.shape!=a[0].shape or x.dtype!=np.float64 or not np.isfinite(x).all() for x in a): raise ValueError('invalid observations')
    if any(np.any(x<=0) for x in a): raise ValueError('positive values required for log reflectometry')
    return a
