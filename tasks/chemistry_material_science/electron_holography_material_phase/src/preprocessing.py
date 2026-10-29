"""Observation validation; acquisition constants match the public README."""
from pathlib import Path
import numpy as np

PIXEL_SIZE_NM = 0.9197516441345215
SHAPE = (1, 512, 512)


def load_observations(path: str | Path = 'data/raw_data.npz') -> dict[str, np.ndarray]:
    """Read object and independently measured vacuum intensity in detector counts."""
    with np.load(path, allow_pickle=False) as f:
        result = {}
        for key in ('hologram', 'vacuum'):
            if key not in f:
                raise ValueError(f'Missing observation: {key}')
            x = f[key]
            if x.shape != SHAPE or x.dtype.kind not in 'iuf' or not np.isfinite(x).all() or np.any(x < 0):
                raise ValueError(f'Invalid observation {key}: expected finite nonnegative {SHAPE}')
            result[key] = x.astype(np.float64)
    return result
