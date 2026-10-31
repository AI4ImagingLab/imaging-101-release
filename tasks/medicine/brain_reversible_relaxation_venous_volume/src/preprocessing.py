"""Measured ASE input and fixed observation-only reporting region."""
from pathlib import Path
import numpy as np
CONDITIONS=('CSF null','No null','GM null','WM null')
TAU=np.arange(-28,65,4,dtype=np.float64)*1e-3
SPACING_MM=(3.75,3.75,5.0)
SHAPE=(1,4,64,64,10,24)

def load_observations(path):
    with np.load(Path(path),allow_pickle=False) as a: v=a['measurements'].copy()
    if v.shape!=SHAPE or v.dtype!=np.float32 or not np.isfinite(v).all() or np.any(v<0):
        raise ValueError('Expected finite nonnegative float32 ASE measurements of shape '+str(SHAPE))
    return v[0]

def observation_mask(signals):
    """Return (4,64,64,10) mask using only public magnitude data."""
    if signals.shape!=(4,64,64,10,24): raise ValueError('Unexpected signal shape')
    x,y,z=np.indices((64,64,10))
    central=(((x-32)/15)**2+((y-32)/18)**2<1)&(z>=2)&(z<=7)
    se=signals[...,7]
    threshold=.5*np.percentile(se,95,axis=(1,2,3))
    return central[None]&(se>threshold[:,None,None,None])&np.all(signals>0,axis=-1)

def log_measurements(signals):
    """Original convention: log magnitude; nonfinite logarithms become zero."""
    values=np.asarray(signals,dtype=np.float64)
    if np.any(values<0) or not np.isfinite(values).all(): raise ValueError('Invalid magnitudes')
    with np.errstate(divide='ignore'): logs=np.log(values)
    return np.where(np.isfinite(logs),logs,0.)
