"""Preserve the upstream per-plane camera normalization (no learned calibration)."""
from pathlib import Path
import numpy as np
NPLANES=8
SHAPE=(512,512)
PIXEL_SIZE=1.12
WAVELENGTH=0.532
DISTANCES=np.arange(300.,651.,50.)
def load_data(path):
    with np.load(Path(path),allow_pickle=False) as z:
        a=z['measurements']
    if a.shape != (1,8,512,512) or a.dtype != np.uint16:
        raise ValueError('measurements must have shape (1,8,512,512), uint16')
    return a[0]
def amplitude_stack(measurements):
    a=np.asarray(measurements).astype('float32')/65535
    low=a.min(axis=(-2,-1),keepdims=True)
    return np.sqrt((a-low)/(1-low)).astype('complex64')
