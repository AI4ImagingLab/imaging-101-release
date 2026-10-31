"""Load public observations without denoising, normalization or target access."""
from pathlib import Path
import numpy as np
def load_observations(path):
    with np.load(Path(path),allow_pickle=False) as f:
        if set(f.files)!={'intensities','angles'}:raise ValueError('Expected intensities and angles')
        y=f['intensities'];angles=f['angles']
    if y.shape!=(1,32,23,26) or y.dtype!=np.uint16:raise ValueError('Invalid native observation grid/dtype')
    if angles.shape!=(1,32) or angles.dtype!=np.float64 or not np.isfinite(angles).all():raise ValueError('Invalid angle calibration')
    return y,angles
