"""Load measured deflection and apply the selected temporal measurement filter."""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
SHAPE=(1,250,79,97)
PIXEL_SIZE=0.0030065919701586503
SAMPLING_RATE=75000.0
TIME_SIGMA=6.0

def load_observations(path: str | Path = 'data/raw_data.npz') -> np.ndarray:
    """Return complete derived real deflection, float64 (1,250,79,97), meters."""
    with np.load(path,allow_pickle=False) as f:a=f['deflection']
    if a.shape!=SHAPE or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('invalid float64 deflection observations')
    return a

def filter_deflection(deflection: np.ndarray, sigma: float = TIME_SIGMA) -> np.ndarray:
    """Gaussian filter on time axis -3, nearest extension, truncate4, float64.

    No spatial filter/downsample occurs in the runtime: it is already part of
    the optically measured, published-example deflection observation provenance.
    """
    a=np.asarray(deflection,dtype=np.float64)
    if sigma<0:raise ValueError('sigma must be nonnegative')
    return gaussian_filter1d(a,sigma=sigma,axis=-3,mode='nearest',truncate=4.0) if sigma else a.copy()
