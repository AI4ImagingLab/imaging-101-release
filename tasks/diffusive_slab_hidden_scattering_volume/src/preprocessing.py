"""Native photon-count loading and the author's normalized temporal resampling."""
from pathlib import Path
import numpy as np

RAW_SHAPE=(512,32,32)

def load_observations(path: str | Path) -> np.ndarray:
    """Return all512 native count bins; no hidden arrays or inferred metadata."""
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!={'photon_counts'}:raise ValueError('Expected photon_counts only')
        x=f['photon_counts']
    if x.shape!=(1,)+RAW_SHAPE or x.dtype!=np.float32:raise ValueError('Expected float32(1,512,32,32)')
    if not np.isfinite(x).all() or np.any(x<0):raise ValueError('Invalid photon counts')
    return x[0].copy()

def preprocess(counts: np.ndarray) -> np.ndarray:
    """Interpolate first417 bins onto128 inclusive points, normalize global peak."""
    a=np.asarray(counts)
    if a.shape!=RAW_SHAPE or a.dtype.kind not in 'fiu' or not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('Expected finite nonnegative512x32x32 observations')
    result=np.empty((128,32,32),dtype=np.float64)
    source=np.linspace(0,1,417); target=np.linspace(0,1,128)
    for i in range(32):
        for j in range(32):result[:,i,j]=np.interp(target,source,a[:417,i,j])
    maximum=result.max()
    if maximum<=0:raise ValueError('No observed photons')
    result/=maximum
    return result.astype(np.float32)
