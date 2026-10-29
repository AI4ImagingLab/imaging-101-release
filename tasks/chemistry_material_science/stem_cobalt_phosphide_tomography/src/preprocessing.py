"""The public data are fixed, integrated detector pixels from existing measurements."""
from pathlib import Path
import numpy as np
PIXEL_NM=6.39
NITER=100
STEP=0.0001
SHAPE=(128,128,128)
def load_data(path):
    with np.load(Path(path),allow_pickle=False) as z:d=z['measurements'];a=z['angles']
    if d.shape!=(1,74,128,128) or d.dtype!=np.float64:raise ValueError('measurements must be float64 (1,74,128,128)')
    if a.shape!=(1,74) or a.dtype!=np.float64:raise ValueError('angles must be float64 (1,74)')
    if not np.isfinite(d).all() or not np.isfinite(a).all():raise ValueError('nonfinite observations')
    return d[0].transpose(2,1,0).copy(),a[0].copy()
