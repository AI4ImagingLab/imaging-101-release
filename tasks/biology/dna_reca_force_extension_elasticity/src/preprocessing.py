"""No inferred parameter is used to calibrate the measured force/extension."""
from pathlib import Path
import numpy as np
SHAPES={'force_control':1318,'distance_control':1318,'force_reca':1329,'distance_reca':1329}
def load_data(path='data/raw_data.npz'):
    with np.load(Path(path),allow_pickle=False) as f:
        data={k:f[k][0].copy() for k in SHAPES}
    for k,a in data.items():
        if a.dtype!=np.float64 or a.shape!=(SHAPES[k],) or not np.isfinite(a).all():raise ValueError('invalid '+k)
    for name in ['control','reca']:data['mask_'+name]=data['force_'+name]<=30
    return data
