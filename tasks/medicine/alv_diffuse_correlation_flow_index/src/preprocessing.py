"""Count-weighted ALV correlations and the original contiguous fit support."""
from pathlib import Path
import numpy as np
def load_data(path='data/raw_data.npz'):
    with np.load(Path(path),allow_pickle=False) as z:return {k:z[k][0] for k in z.files}
def prepare(data):
    g2=data['g2'];counts=data['countrate'];tau=data['lag_times']
    weighted=np.sum(g2*counts[:,:,None],axis=1)/np.sum(counts,axis=1)[:,None]
    masks=np.zeros_like(weighted,dtype=bool)
    first=np.searchsorted(tau,1e-7,side='right');last_time=np.searchsorted(tau,1e-3,side='right')
    for i,row in enumerate(weighted):
        crossing=np.where(row>1.13)[0]
        last=min(last_time,crossing[-1]+1) if crossing.size else last_time
        masks[i,first:last]=True
    return weighted,masks
