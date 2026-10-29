from pathlib import Path
import numpy as np
def load_observation(path: str | Path) -> np.ndarray:
    """Return complete native RGB array, preserving JPEG/255 source units."""
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'observation'}:raise ValueError('Expected only observation')
        y=z['observation']
    if y.shape!=(1,1080,1440,3) or y.dtype!=np.float64:raise ValueError('Expected real float64(1,1080,1440,3)')
    if not np.isfinite(y).all() or np.any(y<0) or np.any(y>1):raise ValueError('Observation must be finite in[0,1]')
    return y[0]
