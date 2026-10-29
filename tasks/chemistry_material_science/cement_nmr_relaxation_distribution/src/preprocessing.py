"""Public NPZ reader. Numerical preprocessing is defined below."""
import numpy as np
def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as f:
        data={k:f[k].copy() for k in f.files}
    if any(v.shape[0]!=1 or not np.isfinite(v).all() for v in data.values()):
        raise ValueError('Expected finite batch-first measurements, batch size 1')
    return {k:v[0] for k,v in data.items()}

def phase_echo(echo):
    """Author four-echo phase estimate (echo indices 1..4), then discard echo 0."""
    a=np.asarray(echo).real;b=np.asarray(echo).imag
    theta=np.mean(np.arctan(b[:,1:5]/a[:,1:5])+np.where(a[:,1:5]<0,np.pi,0),axis=1)
    return (a*np.cos(theta[:,None])+b*np.sin(theta[:,None]))[:,1:]
