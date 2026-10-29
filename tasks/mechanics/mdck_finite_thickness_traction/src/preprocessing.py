"""Measured displacement preparation, preserving pyTFM conventions."""
import numpy as np
PIXEL_SIZE_UM=.201
GRID_PITCH_UM=8.040000000000001
YOUNG_PA=49000.
POISSON=.49
THICKNESS_UM=300.

def load_data(path='data/raw_data.npz'):
    """Return measured displacement (2,48,49), in original image pixels."""
    with np.load(path,allow_pickle=False) as d:u=d['displacement'][0]
    if u.shape!=(2,48,49) or u.dtype!=np.float64 or not np.isfinite(u).all():
        raise ValueError('Expected finite float64 displacement (1,2,48,49)')
    return u

def prepare_displacement(displacement,pixel_size=PIXEL_SIZE_UM):
    """Subtract each component's mean, convert pixels→µm, pad bottom/right.

    Input (2,H,W); return (2,N,N), N smallest even integer >=max(H,W).
    Data occupy final H rows and final W columns, matching finite pyTFM.
    """
    u=np.asarray(displacement)
    if u.ndim!=3 or u.shape[0]!=2:raise ValueError('Expected two displacement components')
    h,w=u.shape[1:];n=max(h,w);n+=n%2
    padded=np.zeros((2,n,n),dtype=np.float64)
    padded[:,-h:,-w:]=(u-u.mean(axis=(1,2),keepdims=True))*pixel_size
    return padded
