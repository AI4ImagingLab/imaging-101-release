"""The input is existing author-processed, amplitude/phase-composed real pressure.

No input normalization or individual-speaker reprocessing is performed here.
"""
import numpy as np
def load_pressure(path):
    """Return the public complex64 pressure matrix, shape (9,12), in author units U."""
    with np.load(path,allow_pickle=False) as z:p=z['pressure']
    if p.shape!=(1,9,12) or p.dtype!=np.complex64 or not np.isfinite(p).all():
        raise ValueError('pressure must be finite complex64 (1,9,12)')
    if not np.linalg.norm(p):raise ValueError('pressure is identically zero')
    return p[0]
