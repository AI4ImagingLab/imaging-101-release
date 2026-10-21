"""Public NPZ reader. Numerical preprocessing is defined below."""
import numpy as np
def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as f:
        data={k:f[k].copy() for k in f.files}
    if any(v.shape[0]!=1 or not np.isfinite(v).all() for v in data.values()):
        raise ValueError('Expected finite batch-first measurements, batch size 1')
    return {k:v[0] for k,v in data.items()}

def prepare(time,echo):
    """Analytical imaginary-energy minimization; positive real mean, peak normalization."""
    a=echo.real;b=echo.imag
    phi=.5*np.arctan2(np.sum(a*b),.5*np.sum(b*b-a*a))+np.pi/2
    y=(echo*np.exp(1j*phi)).real
    if y.mean()<0:y=-y
    return time-time[0]+.1,y/y.max()
