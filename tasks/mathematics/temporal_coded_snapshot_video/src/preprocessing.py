"""Read author measured full-resolution observations and independent coded masks."""
from pathlib import Path
import numpy as np

def load_data(path=Path('data/raw_data.npz')):
    shapes={'mask':(1,14,256,256),'observations':(1,15,256,256)}
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(shapes):raise ValueError('Require exact visible observation/calibration keys')
        data={k:z[k].copy() for k in shapes}
    for k,value in data.items():
        if value.shape!=shapes[k] or value.dtype!=np.float64 or not np.isfinite(value).all():raise ValueError('Invalid '+k)
    if np.any(data['mask']<0) or np.any(data['mask']>1):raise ValueError('Invalid effective mask transmission')
    if np.any(np.sum(data['mask'][0]**2,axis=0)<=0):raise ValueError('Bound mask has an unobserved sensor pixel')
    return {k:v[0] for k,v in data.items()}
