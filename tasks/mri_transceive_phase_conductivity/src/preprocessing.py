"""Native real observations and source-preserving repeat averaging."""
from pathlib import Path
import numpy as np
from scipy.ndimage import minimum_filter

NATIVE_SHAPE = (96, 96, 80)  # z, y, x; original library stores x fastest.
REPEATS = 25


def load_observations(path):
    """Return phase/magnitude float32 [repeat,z,y,x], and bool source mask."""
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {'transceive_phase', 'transmit_magnitude', 'observation_mask', 'reconstruction_support'}
        if not required.issubset(archive.files):
            raise ValueError('Missing observation or magnitude/mask array')
        phase = archive['transceive_phase']
        magnitude = archive['transmit_magnitude']
        mask = archive['observation_mask']
        support = archive['reconstruction_support']
    for name, array in [('transceive_phase', phase), ('transmit_magnitude', magnitude)]:
        if array.shape != (1, REPEATS, *NATIVE_SHAPE) or array.dtype != np.float32:
            raise ValueError(name + ' must have the declared native float32 shape')
    if mask.shape != (1, *NATIVE_SHAPE) or mask.dtype != np.bool_:
        raise ValueError('observation_mask must have the declared bool shape')
    if support.shape != mask.shape or support.dtype != np.bool_:
        raise ValueError('reconstruction_support must have the declared bool shape')
    if not np.array_equal(support[0], minimum_filter(mask[0], size=11, mode='constant', cval=False)):
        raise ValueError('Public output support disagrees with available11cube stencil')
    for array in [phase[0], magnitude[0]]:
        if not np.all(np.isfinite(array[:, mask[0]])):
            raise ValueError('Nonfinite observation inside public source support')
        if not np.all(np.isnan(array[:, ~mask[0]])):
            raise ValueError('Outside source support must retain native NaNs')
    if np.any(magnitude[0, :, mask[0]] <= 0):
        raise ValueError('Transmit magnitude must be positive inside support')
    return phase[0], magnitude[0], mask[0]


def average_repeats(values):
    """Average [repeat,...] in float64, summing each input divided by count.

    This division-before-accumulation order is the dataset author's main.cc.
    NaN exterior is preserved. No spatial resampling or phase unwrapping.
    """
    values = np.asarray(values)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError('Need at least one repeat with a spatial array')
    result = np.zeros(values.shape[1:], dtype=np.float64)
    for sample in values:
        result += sample.astype(np.float64) / values.shape[0]
    return result
