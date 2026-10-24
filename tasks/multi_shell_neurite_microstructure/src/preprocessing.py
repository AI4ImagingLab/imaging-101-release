"""Validated native observations; all supplied spatial/acquisition axes remain."""
from pathlib import Path
import numpy as np

def load_data(path):
    with np.load(Path(path),allow_pickle=False) as f:
        expected={'diffusion_signal','bvalues','gradient_directions','observation_mask'}
        if set(f.files)!=expected:raise ValueError('Unexpected raw NPZ keys')
        data={key:f[key] for key in expected}
    signal=data['diffusion_signal'];b=data['bvalues'];g=data['gradient_directions'];mask=data['observation_mask']
    if signal.shape!=(1,96,96,19,496) or signal.dtype!=np.int16:raise ValueError('Native diffusion signal must be int16(1,96,96,19,496)')
    if b.shape!=(1,496) or b.dtype!=np.float64 or g.shape!=(1,496,3) or g.dtype!=np.float64:raise ValueError('Invalid diffusion encoding arrays')
    if mask.shape!=(1,96,96,19) or mask.dtype!=np.bool_:raise ValueError('Invalid fixed observation support')
    if any(not np.isfinite(a).all() for a in data.values()) or np.any(b<0):raise ValueError('Nonfinite data or negative b values')
    if not np.array_equal(np.unique(b),np.arange(0,3001,200)):raise ValueError('Acquisition shell values changed')
    return data
