"""Validate the native, uncalibrated VNA sweep."""
import numpy as np

def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as archive:
        f=archive['frequency'][0]
        z=archive['scattering'][0]
    if f.dtype!=np.float64 or z.dtype!=np.complex128 or f.shape!=z.shape or f.ndim!=1:
        raise ValueError('Expected aligned float64 frequency and complex128 scattering vectors')
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(z)) or not np.all(np.diff(f)>0):
        raise ValueError('Invalid sweep')
    return f,z
