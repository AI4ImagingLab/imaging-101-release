"""Closed-form parameter mapping on the supplied brain mask."""
import numpy as np
from .preprocessing import load_data
from .physics_model import linearized_fit
def reconstruct(path):
    """Return batch-first float64 T1(ms) and M0 maps; zero outside mask."""
    s,b,m=load_data(path);t1=np.zeros(m.shape);m0=np.zeros(m.shape)
    t1[m],m0[m]=linearized_fit(s[m],b[m])
    return {'reconstruction':t1[None],'equilibrium_magnetization':m0[None]}
def reconstruct_uncorrected(path):
    """Fresh scientifically relevant comparison assuming unit transmit field."""
    s,_,m=load_data(path);out=np.zeros(m.shape);out[m]=linearized_fit(s[m])[0]
    return out[None]
