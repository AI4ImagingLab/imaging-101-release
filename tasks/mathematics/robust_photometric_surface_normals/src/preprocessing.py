"""Strict loading of author-provided existing rendered observations."""
from pathlib import Path
import numpy as np
SHAPE=(256,256)
N_LIGHTS=50
def load_observations(path=Path('data/raw_data.npz')):
    with np.load(path,allow_pickle=False) as z:
        y=z['measurements'];lights=z['light_directions'];support=z['support']
    if y.shape!=(1,N_LIGHTS,*SHAPE) or y.dtype!=np.float64:raise ValueError('Invalid observations')
    if lights.shape!=(1,N_LIGHTS,3) or lights.dtype!=np.float64:raise ValueError('Invalid illumination')
    if support.shape!=(1,*SHAPE) or support.dtype!=np.bool_:raise ValueError('Invalid observed support')
    if not np.isfinite(y).all() or not np.isfinite(lights).all():raise ValueError('Nonfinite observations')
    return y[0],lights[0],support[0]
def measurement_matrix(measurements):
    """(light,y,x) to (pixel,light) in C-order, no intensity renormalization."""
    y=np.asarray(measurements)
    if y.ndim!=3 or not np.isfinite(y).all():raise ValueError('Expected finite light,y,x observations')
    return y.reshape(y.shape[0],-1).T
