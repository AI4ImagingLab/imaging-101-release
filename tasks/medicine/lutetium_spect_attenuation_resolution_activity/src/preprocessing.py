"""Lossless count inputs, independent CT calibration, TEW scatter estimate."""
from pathlib import Path
import numpy as np
import torch

WINDOW_BOUNDS = ((187.19999694824,228.80000305176),
                 (166.39999389648,187.19999694824),
                 (228.80000305176,249.60000610352))
SHAPES = {'energy_windows':(1,3,96,128,128),'attenuation':(1,128,128,128),
          'angles':(1,96),'radii':(1,96)}
DTYPES = {'energy_windows':np.dtype('uint16'),'attenuation':np.dtype('float32'),
          'angles':np.dtype('float32'),'radii':np.dtype('float64')}


def load_data(path):
    with np.load(Path(path),allow_pickle=False) as z:
        arrays = {k:z[k] for k in SHAPES}
    for k,a in arrays.items():
        if a.shape!=SHAPES[k] or a.dtype!=DTYPES[k] or not np.isfinite(a).all():
            raise ValueError('Invalid observation/calibration '+k)
    if np.any(arrays['attenuation']<0) or np.any(arrays['radii']<=0):
        raise ValueError('Nonphysical calibration')
    return {k:a[0] for k,a in arrays.items()}


def scatter_estimate(energy_windows, device='cpu'):
    counts = torch.as_tensor(energy_windows,dtype=torch.float32,device=device)
    peak,lower,upper = [hi-lo for lo,hi in WINDOW_BOUNDS]
    return (counts[1]/lower*.5 + counts[2]/upper*.5)*peak
