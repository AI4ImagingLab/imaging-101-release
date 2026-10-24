"""Validate normalized encoded RGB observations without hidden input access."""
from pathlib import Path
import numpy as np

def load_observations(path=Path('data/raw_data.npz')):
    with np.load(path,allow_pickle=False) as data:
        image=data['observations'][0].copy()
        airlight=data['airlight'][0].copy()
    if image.ndim!=3 or image.shape[-1]!=3 or image.dtype!=np.float64:
        raise ValueError('observations must be float64 H x W x 3')
    if airlight.shape!=(3,) or airlight.dtype!=np.float64 or not np.isfinite(airlight).all() or np.any((airlight<=0)|(airlight>1)):raise ValueError('invalid RGB airlight')
    if not np.isfinite(image).all() or np.any((image<0)|(image>1)):raise ValueError('invalid normalized RGB observations')
    return image,airlight
