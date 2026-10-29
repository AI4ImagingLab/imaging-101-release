"""Validated observations; no design gate or fitted calibration is loaded."""
from pathlib import Path
import numpy as np

def load_data(path='data/raw_data.npz'):
    """Return unbatched complex expectations(2,4,50), angles(50), weights(50)."""
    with np.load(Path(path), allow_pickle=False) as z:
        d = {k: z[k][0].copy() for k in z.files}
    expected = {'expectation': (2,4,50), 'polar_angle': (50,), 'azimuth': (50,), 'quadrature_weight': (50,)}
    if set(d) != set(expected):
        raise ValueError('Unexpected public data keys')
    for k, shape in expected.items():
        if d[k].shape != shape or not np.isfinite(d[k]).all():
            raise ValueError(f'Invalid {k}')
    return d

def droplets(expectation):
    """Combine scalar/vector measured ranks; return (X,Y,Z,I) droplets(4,N)."""
    e = np.asarray(expectation)
    f = (e[0] + np.sqrt(3)*e[1])/(4*np.sqrt(2*np.pi))
    return f[[1,2,3,0]] * np.array([1j,1j,1j,1])[:,None]
