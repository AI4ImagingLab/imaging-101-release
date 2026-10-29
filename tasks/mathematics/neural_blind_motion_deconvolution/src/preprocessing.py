"""Validated public observation loading and batch-axis handling."""
from pathlib import Path
import numpy as np

IMAGE_SHAPE = (753, 668)
KERNEL_SIZE = 51
QUANTIZATION_STEP = 1.0 / 255.0


def load_observation(path='data/raw_data.npz'):
    """Read dimensionless float32 luminance and return a fresh (753,668) array."""
    with np.load(Path(path), allow_pickle=False) as data:
        if set(data.files) != {'observation'}:
            raise ValueError('raw archive must contain exactly observation')
        array = data['observation']
    if array.shape != (1, *IMAGE_SHAPE) or array.dtype != np.float32:
        raise ValueError('observation must have shape (1,753,668) and dtype float32')
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError('observation must be finite normalized luminance in [0,1]')
    return array[0].copy()
