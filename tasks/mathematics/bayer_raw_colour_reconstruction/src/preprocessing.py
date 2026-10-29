"""Unchanged sensor ADUs and explicit black/white-level normalization."""
from pathlib import Path
import numpy as np
from .physics_model import PATTERN,WHITE_LEVEL,SHAPE,cfa_masks

def load_observations(path=Path('data/raw_data.npz')):
    """Return unbatched uint16 CFA, float64 black[4], white-balance[4]."""
    with np.load(path,allow_pickle=False) as z:
        raw=z['sensor_mosaic'];black=z['black_levels'];wb=z['white_balance']
    if raw.shape!=(1,*SHAPE) or raw.dtype!=np.uint16:raise ValueError('Invalid sensor mosaic')
    if black.shape!=(1,4) or black.dtype!=np.float64 or wb.shape!=(1,4) or wb.dtype!=np.float64:raise ValueError('Invalid calibration')
    if not all(np.isfinite(x).all() for x in [raw,black,wb]):raise ValueError('Nonfinite observations')
    if np.any(black<0) or np.any(black>=WHITE_LEVEL):raise ValueError('Invalid black levels')
    return raw[0],black[0],wb[0]

def normalize_mosaic(raw,black_levels,white_level=WHITE_LEVEL,pattern=PATTERN):
    """Normalize each measured ADU; no clipping, crop, rescale or rotation.

    Black levels follow LibRaw colour descriptors RGBG, not Bayer grid order.
    """
    a=np.asarray(raw);b=np.asarray(black_levels,dtype=np.float64)
    if a.ndim!=2 or b.shape!=(4,) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid normalization inputs')
    if np.any(b>=white_level):raise ValueError('Nonpositive sensor range')
    masks=cfa_masks(a.shape,pattern)
    # This camera stores equal31ADU for both greens; preserve exact descriptor values.
    if b[1]!=b[3]:raise ValueError('This calibrated task requires identical green black levels')
    bm=sum(b[i]*m for i,m in enumerate(masks))
    return (a.astype(np.float64)-bm)/(white_level-bm)
