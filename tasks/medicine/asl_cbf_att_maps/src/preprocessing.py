"""Validate the public batch-first observation and calibration arrays."""
from pathlib import Path
import numpy as np

def load_observations(path=Path('data/raw_data.npz')):
    """Load exactly magnetization float32 (1,7,5,35,35), M0 float32 and
    binary mask uint8 (1,5,35,35); no clipping/rescaling or hidden inputs.
    """
    specs={'magnetization':((1,7,5,35,35),np.dtype('float32')),'m0':((1,5,35,35),np.dtype('float32')),'mask':((1,5,35,35),np.dtype('uint8'))}
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(specs):raise ValueError('Unexpected public observation keys')
        data={k:z[k].copy() for k in specs}
    for key,(shape,dtype) in specs.items():
        if data[key].shape!=shape or data[key].dtype!=dtype or not np.isfinite(data[key]).all():raise ValueError(f'Invalid {key} shape, dtype or values')
    if not np.isin(data['mask'],[0,1]).all():raise ValueError('Mask must be binary')
    if np.any(data['m0'][data['mask']>0]<=0):raise ValueError('Masked M0 must be positive')
    return data
