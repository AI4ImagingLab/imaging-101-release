"""Load the complete measured30-chronometer cohort without BAO substitutions."""
from pathlib import Path
import numpy as np

def load_observations(path=Path('data/raw_data.npz')):
    with np.load(path,allow_pickle=False) as f:
        z,h,error=(f[k][0].copy() for k in ['redshift','hubble_observation','hubble_error'])
    if any(v.shape!=(30,) for v in [z,h,error]) or not np.all(np.isfinite([z,h,error])):raise ValueError('Thirty finite chronometer measurements required')
    if np.any(error<=0) or np.any(z<0):raise ValueError('Positive observational errors and nonnegative redshift required')
    return z,h,error

def reconstruction_grid():
    return np.linspace(0.,1.965,200,dtype=np.float64)
