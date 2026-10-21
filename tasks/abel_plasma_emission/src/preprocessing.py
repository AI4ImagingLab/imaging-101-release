"""Acquisition constants and lossless observation loading."""
from pathlib import Path
import numpy as np

DR = 0.025  # mm
DZ = 0.1  # mm
NOISE_STD = 0.002  # relative radiance units
SHAPE = (1, 65, 257)


def load_projection(path: str | Path) -> np.ndarray:
    """Load float64 (batch, z, impact-parameter) radiance without modification."""
    with np.load(path, allow_pickle=False) as data:
        if 'projection' not in data:
            raise ValueError('Missing projection array')
        projection = data['projection']
    if projection.shape != SHAPE or projection.dtype != np.float64:
        raise ValueError(f'projection must be float64 with shape {SHAPE}')
    if not np.isfinite(projection).all():
        raise ValueError('projection contains nonfinite values')
    return projection
