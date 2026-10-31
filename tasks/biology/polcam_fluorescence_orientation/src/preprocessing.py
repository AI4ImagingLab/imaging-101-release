"""Lossless loading and micro-polarizer channel extraction."""
from pathlib import Path
import numpy as np
RAW_SHAPE = (1, 116, 298, 298)
POLARIZER_ORDER = (-45, 0, 90, 45)
OFFSET_ADU = 0.0

def load_observations(path: str | Path = 'data/raw_data.npz') -> np.ndarray:
    """Return uint16 intensity (batch, plane, row, column) in ADU."""
    with np.load(path, allow_pickle=False) as archive:
        raw = archive['intensity']
    if raw.shape != RAW_SHAPE or raw.dtype != np.uint16:
        raise ValueError('intensity must be uint16 with shape '+str(RAW_SHAPE))
    return raw

def split_channels(raw: np.ndarray, offset: float = OFFSET_ADU) -> np.ndarray:
    """Return float64 (..., 4, row/2, col/2), channels 0,45,90,135 deg.

    Adjacent 2x2 detector samples form one effective superpixel. No interpolation,
    intensity clipping, rotation or smoothing is applied. Raw row axis points down.
    """
    a = np.asarray(raw, dtype=np.float64) - offset
    if a.ndim < 2 or a.shape[-2] % 2 or a.shape[-1] % 2:
        raise ValueError('last two dimensions must be even')
    if not np.isfinite(a).all():
        raise ValueError('nonfinite observations')
    return np.stack((a[..., ::2, 1::2], a[..., 1::2, 1::2],
                     a[..., 1::2, ::2], a[..., ::2, ::2]), axis=-3)
