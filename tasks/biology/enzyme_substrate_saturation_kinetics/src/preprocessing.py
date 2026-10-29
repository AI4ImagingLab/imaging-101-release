"""Public initial rates are already converted to micromolar/minute."""
from pathlib import Path
import numpy as np
def load_data(path='data/raw_data.npz'):
    with np.load(Path(path),allow_pickle=False) as a:
        s=a['substrate'];v=a['rates']
    if s.shape!=(1,10) or v.shape!=(1,8,10): raise ValueError('Invalid public array shapes')
    if s.dtype!=np.float64 or v.dtype!=np.float64: raise ValueError('Expected float64')
    if not np.isfinite(s).all() or not np.isfinite(v).all() or np.any(s<=0) or np.any(v<=0): raise ValueError('Expected finite positive observations')
    if np.any(np.diff(s[0])<=0): raise ValueError('Concentrations must increase')
    return s[0].copy(),v[0].copy()
