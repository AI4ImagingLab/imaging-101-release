"""Load measured FLASH images and define an observation-only analysis region."""
from pathlib import Path
import numpy as np
SHAPE=(128,128,96)
def load_data(path='data/raw_data.npz'):
    with np.load(Path(path),allow_pickle=False) as f:
        if any(f[k].shape!=(1,)+SHAPE for k in ('PDw','T1w','MTw')): raise ValueError('Invalid batch-first FLASH shape')
        d={k:f[k][0].copy() for k in ('PDw','T1w','MTw')}
    for a in d.values():
        if a.shape!=SHAPE or a.dtype!=np.float64 or not np.isfinite(a).all(): raise ValueError('Invalid FLASH image')
    return d
def analysis_mask(data):
    p,t,m=(data[k] for k in ('PDw','T1w','MTw'))
    x,y,z=np.indices(SHAPE)
    central=((x-63)/38)**2+((y-64)/53)**2+((z-48)/30)**2<1
    u=p/(6*np.pi/180);v=t/(20*np.pi/180)
    condition=np.abs(u-v)/(u+v+1e-100)
    return central&(p>100)&(t>100)&(m>50)&(condition>.2)
