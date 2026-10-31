"""Strict public observation loader, retaining all native displacement samples."""
from pathlib import Path
import numpy as np
def load_observations(path):
    with np.load(Path(path),allow_pickle=False) as archive:
        result={}
        for key in ['positions','displacement','displacement_sigma']:
            if key not in archive:raise ValueError('Missing observation '+key)
            a=archive[key]
            if a.shape!=(1,32,32,2) or a.dtype!=np.dtype('float64') or not np.isfinite(a).all():raise ValueError('Malformed observation '+key)
            result[key]=a.copy()
        if np.any(result['displacement_sigma']<=0):raise ValueError('Nonpositive uncertainty')
    return result
