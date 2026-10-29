"""Read real SNSPD photons and independently measured instrument response."""
import numpy as np
def normalize_irf(raw):
    """Peak-zero circular template, float64(N,), unit-sum; raw peak is electronics zero."""
    a=np.asarray(raw,dtype=np.float64)
    if a.ndim!=1 or not np.isfinite(a).all() or (a<0).any():raise ValueError('Invalid nonnegative instrument response')
    with np.errstate(over='ignore'):total=a.sum()
    if not np.isfinite(total) or total<=0:raise ValueError('Invalid instrument-response sum')
    a=np.roll(a,-int(np.argmax(a)));return a/a.sum()
def load_data(path):
    """Return uint32(50,50,10000) photon counts and float64(10000,) normalized IRF."""
    with np.load(path,allow_pickle=False) as z:counts=z['photon_counts'];irf=z['instrument_response']
    if counts.shape!=(1,50,50,10000) or counts.dtype!=np.uint32:raise ValueError('photon_counts must be uint32(1,50,50,10000)')
    if irf.shape!=(1,10000) or irf.dtype!=np.float64:raise ValueError('instrument_response must be float64(1,10000)')
    return counts[0],normalize_irf(irf[0])
