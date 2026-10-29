"""Load author-processed correlation measurements and select reliable lags."""
from pathlib import Path
import numpy as np
N=1.333
ANGLE_DEGREES=173.0
WAVELENGTH_NM=633.0
RADIUS_NM=250.0
TEMPERATURE_K=310.15
KB=1.38e-23
LENGTHS=(188,145,155)
def load_data(path='data/raw_data.npz'):
    """Return two (3,192) float64 arrays, lag in microseconds and g2-1."""
    with np.load(Path(path),allow_pickle=False) as z:
        t,g=z['time_lag'],z['correlation']
    if t.shape!=(1,3,192) or g.shape!=t.shape or t.dtype!=np.float64 or g.dtype!=np.float64:
        raise ValueError('Expected float64 batch-first arrays (1,3,192)')
    if not np.isfinite(t).all() or not np.isfinite(g).all() or not (np.diff(t,axis=-1)>0).all():
        raise ValueError('Finite observations and strictly increasing lags required')
    return t[0],g[0]
def select_lags(t,g):
    """Original rule: skip first 3 lags; stop before first g<0.05, else last."""
    stop=int(np.argmax(g<.05))
    if stop==0: stop=len(g)-1
    return t[3:stop].copy(),g[3:stop].copy()
