"""Public measurement contract and exact upstream crop convention."""
from pathlib import Path
import numpy as np

SPACING = (1.0625, 1.0625, 1.0714285714285714)
B0_T = 2.89362
TE_S = 0.025
GYRO_MHZ_T = 42.58
PHASE_PER_PPM = 2*np.pi*GYRO_MHZ_T*B0_T*TE_S
SHAPE = (1, 160, 160, 160)


def load_data(path='data/raw_data.npz'):
    """Load and validate batch-first phase, magnitude and acquisition mask."""
    with np.load(Path(path), allow_pickle=False) as z:
        expected = {'phase': np.dtype('float64'), 'magnitude': np.dtype('float32'), 'mask': np.dtype('bool')}
        if set(z.files) != set(expected):
            raise ValueError('Expected phase, magnitude and mask only')
        arrays = {}
        for key, dtype in expected.items():
            a = z[key]
            if a.shape != SHAPE or a.dtype != dtype or not np.isfinite(a).all():
                raise ValueError(f'Invalid {key}: expected {SHAPE}, {dtype}, finite')
            arrays[key] = a[0].copy()
    if not arrays['mask'].any() or (arrays['magnitude'] < 0).any():
        raise ValueError('Mask must be nonempty and magnitude nonnegative')
    return arrays


def crop_slices(mask):
    """Bounding box of nonzero mask, extended by two grid cells each side."""
    indices = np.where(mask)
    if not indices[0].size:
        raise ValueError('Empty acquisition mask')
    return tuple(slice(max(0, int(i.min())-2), min(n, int(i.max())+3))
                 for i, n in zip(indices, mask.shape))
