"""Complete native digital film observations and pixel/time matrix order."""
from pathlib import Path
import numpy as np


def load_data(path='data/raw_data.npz'):
    """Return uint8 grayscale frames(T,H,W), ADU, without resampling."""
    with np.load(Path(path),allow_pickle=False) as f:a=f['frames']
    if a.shape!=(1,84,240,320) or a.dtype!=np.uint8:raise ValueError('Expected native uint8 (1,84,240,320) movie')
    return a[0]


def as_matrix(frames):
    """Pixel-by-time float64 matrix, intensity/255; C order y,x pixels."""
    return frames.reshape(len(frames),-1).T.astype(np.float64)/255


def as_movie(matrix,shape):
    """Undo only the documented matrix ordering; shape=(T,H,W)."""
    return matrix.T.reshape((1,*shape))
