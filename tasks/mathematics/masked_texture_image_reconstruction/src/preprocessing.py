"""Load public observations without truth, hidden masks or scale changes."""
from pathlib import Path
import numpy as np
SHAPE=(1,240,320)
def load_observations(path='data/raw_data.npz'):
    with np.load(Path(path),allow_pickle=False) as z:
        if set(z.files)!={'observations','mask'}:raise ValueError('Expected observations and mask')
        y=z['observations'];mask=z['mask']
    if y.shape!=SHAPE or mask.shape!=SHAPE or y.dtype!=np.float64 or mask.dtype!=np.bool_:raise ValueError('Invalid public shapes/dtypes')
    if not np.isfinite(y).all() or np.any(y[~mask]!=0):raise ValueError('Nonfinite or unobserved values must be zero')
    return y[0],mask[0]

