from pathlib import Path
import numpy as np
PERIOD=12.68550106609808
KERNEL=7
AVERAGE=13
SHAPE=(35,35)

def load_data(path):
    with np.load(Path(path),allow_pickle=False) as z:d={k:z[k] for k in ['grid','sample','flat']}
    for a in d.values():
        if a.shape!=(1,468,468) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('public images must be finite float32 (1,468,468)')
    if np.any(d['flat']<=0):raise ValueError('flat exposure must be positive')
    return d['grid'][0]/d['flat'][0],d['sample'][0]/d['flat'][0]
