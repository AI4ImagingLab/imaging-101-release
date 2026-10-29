"""Exact measured frequencies and partial mode normalization."""
from pathlib import Path
import numpy as np
MASS=12.060/386.088
NOMINAL_STIFFNESS=10.
def load_data(path='data/raw_data.npz'):
    with np.load(Path(path),allow_pickle=False) as z:f=z['frequency'];v=z['measured_modes']
    if f.shape!=(1,2) or v.shape!=(1,3,2) or f.dtype!=np.float64 or v.dtype!=np.float64 or not np.isfinite(f).all() or not np.isfinite(v).all():raise ValueError('Expected finite float64 frequency(1,2), measured_modes(1,3,2)')
    return f[0],v[0]
def normalize_modes(measured):
    q=np.argmax(abs(measured),axis=0)
    return measured/measured[q,np.arange(2)],q
