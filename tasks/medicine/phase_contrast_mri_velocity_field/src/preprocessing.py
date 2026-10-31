"""Observation IO and explicit acquisition coordinates; GPL-3.0-or-later."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PUBLIC_SHAPE=(1,27,4,54,67,100)
VENC_M_S=.16
VOXEL_MM=(1.5,100/67,80/54)

def load_observations(path=None):
    """Return public complex64 encoding_images, batch,time,encoding,z,y,x."""
    with np.load(path or ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        if 'encoding_images' not in z:raise ValueError('Missing encoding_images')
        a=z['encoding_images']
    if a.shape!=PUBLIC_SHAPE or a.dtype!=np.complex64 or not np.isfinite(a).all():
        raise ValueError('encoding_images must be finite complex64 '+str(PUBLIC_SHAPE))
    return a

def native_order(images):
    """Validated public layout to contiguous source x,y,z,time,encoding."""
    if images.shape!=PUBLIC_SHAPE or images.dtype!=np.complex64 or not np.isfinite(images).all():
        raise ValueError('Invalid public observation shape/dtype/finite values')
    return np.ascontiguousarray(images[0].transpose(4,3,2,0,1))
