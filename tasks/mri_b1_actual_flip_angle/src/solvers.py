"""Complete native-volume AFI reconstruction."""
import numpy as np
from .preprocessing import load_data
from .physics_model import invert_afi
def reconstruct(path):
    """Return {'reconstruction': float64 (1,88,128,32)} transmit ratio."""
    signal,mask=load_data(path);field=np.zeros(mask.shape,dtype=np.float64)
    field[mask]=invert_afi(signal[mask])
    return {'reconstruction':field[None]}
