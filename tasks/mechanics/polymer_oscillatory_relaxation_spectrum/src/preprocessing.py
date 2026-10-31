"""Public NPZ reader. Numerical preprocessing is defined below."""
import numpy as np
def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as f:
        data={k:f[k].copy() for k in f.files}
    if any(v.shape[0]!=1 or not np.isfinite(v).all() for v in data.values()):
        raise ValueError('Expected finite batch-first measurements, batch size 1')
    return {k:v[0] for k,v in data.items()}

def prepare(data,n_resample=100):
    w,index=np.unique(data['angular_frequency'],return_index=True)
    gp=data['storage_modulus'][index];gpp=data['loss_modulus'][index]
    wr=np.geomspace(w.min(),w.max(),n_resample)
    return wr,np.r_[np.interp(wr,w,gp),np.interp(wr,w,gpp)],np.ones(2*n_resample)
