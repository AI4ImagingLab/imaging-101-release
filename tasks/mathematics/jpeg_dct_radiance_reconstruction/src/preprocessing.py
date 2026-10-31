"""Load the original integer coefficient and quantization arrays."""
from pathlib import Path
import numpy as np
def load_observations(path):
    with np.load(Path(path),allow_pickle=False) as z:
        a=z['coefficients'];q=z['quantization']
        if a.shape!=(1,3,54,80,8,8) or a.dtype!=np.int16:raise ValueError('invalid coefficients')
        if q.shape!=(1,3,8,8) or q.dtype!=np.uint16 or (q==0).any():raise ValueError('invalid quantization')
        return a[0].copy(),q[0].copy()
