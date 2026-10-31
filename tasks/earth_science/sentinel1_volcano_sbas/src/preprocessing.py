"""Observation loading and acquisition constants; no upstream-library imports."""
from pathlib import Path
import numpy as np

SHAPE = (450, 600)
REFERENCE = (76, 156)
WAVELENGTH = 0.05546576
STARTING_RANGE = 799547.2189154713
RANGE_SPACING = 34.94343172072985
AZIMUTH_SPACING = 78.08212603857966
STEPS = (20170910, 20180613)
EXPECTED = {
    'phase': ((288, *SHAPE), np.dtype('float32')),
    'coherence': ((288, *SHAPE), np.dtype('float32')),
    'components': ((288, *SHAPE), np.dtype('uint8')),
    'dates': ((98,), np.dtype('int32')),
    'pairs': ((288, 2), np.dtype('int16')),
    'baseline': ((98,), np.dtype('float32')),
    'atmospheric_delay': ((98, *SHAPE), np.dtype('float32')),
    'incidence': (SHAPE, np.dtype('float32')),
    'latitude': (SHAPE, np.dtype('float64')),
    'longitude': (SHAPE, np.dtype('float64')),
    'analysis_mask': (SHAPE, np.dtype('bool')),
}


def load_data(path='data/raw_data.npz'):
    """Validate the public batch-first NPZ; return independent arrays without batch."""
    result = {}
    with np.load(Path(path), allow_pickle=False) as archive:
        for key, (shape, dtype) in EXPECTED.items():
            value = archive[key]
            if value.shape != (1, *shape) or value.dtype != dtype:
                raise ValueError(f'{key}: expected {(1, *shape)} {dtype}, got {value.shape} {value.dtype}')
            if not np.isfinite(value).all():
                raise ValueError(f'{key}: nonfinite observation')
            result[key] = value[0].copy()
    return result


def reference_atmosphere(delay, reference=REFERENCE):
    """Re-reference the independent atmospheric LOS product in space and time."""
    value = np.array(delay, dtype=np.float32, copy=True)
    value -= value[:, reference[0], reference[1]][:, None, None]
    value -= value[0:1]
    return value
