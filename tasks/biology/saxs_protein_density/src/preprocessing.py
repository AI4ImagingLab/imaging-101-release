"""Bound real lysozyme scattering observations; no hidden imports."""
from pathlib import Path
import numpy as np
SHAPE=(1,485)
SIZE=64
SIDE=240.0
VOXEL=3.75
DMAX=40.0

def load_observations(path: str | Path='data/raw_data.npz') -> dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as f:data={key:f[key] for key in ['q','intensity','sigma','smoothed_intensity','measured_mask']}
    for key,a in data.items():
        if a.shape!=SHAPE:raise ValueError('required observation shape '+str(SHAPE))
        if key=='measured_mask':
            if a.dtype!=np.bool_:raise ValueError('mask must be boolean')
        elif a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('finite float64 observation required')
    if np.any(np.diff(data['q'][0])<=0) or np.any(data['sigma']<=0) or np.any(data['smoothed_intensity']<=0):raise ValueError('invalid scattering samples')
    return data
