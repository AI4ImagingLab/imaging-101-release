"""Observation-only normalization, preserving original RGB and arithmetic."""
from pathlib import Path
import numpy as np

KEYS={'pattern','white','left_color','right_color','left_depth','right_depth'}

def load_observations(path):
    """Load the six public batched arrays, with finite numeric validation."""
    with np.load(Path(path),allow_pickle=False) as f:
        if set(f.files)!=KEYS:raise ValueError('Incorrect public observation/calibration keys')
        raw={k:f[k] for k in f.files}
    shapes={'pattern':(1,3,650,650,3),'white':(1,3,650,650,3),'left_color':(1,3,650,650,25),'right_color':(1,3,25,749),'left_depth':(1,650,650,25),'right_depth':(1,25,749)}
    for k,a in raw.items():
        dtype=np.dtype('uint16' if k in ('pattern','white') else 'float32')
        if a.shape!=shapes[k] or a.dtype!=dtype or not np.isfinite(a).all():raise ValueError('Invalid public array '+k)
    return raw

def normalize_observations(pattern,white,threshold=.05):
    """Return float32 normalized RGB and input-only support for FxHxWx3 images.

    The threshold uses the minimum white channel and its per-frame global
    maximum. uint16 operands divide in float64 before assignment to float32.
    There is no ambient capture in the dynamic acquisition.
    """
    p=np.asarray(pattern);w=np.asarray(white)
    if p.shape!=w.shape or p.ndim!=4 or p.shape[-1]!=3:raise ValueError('Matched frame/row/column/RGB arrays required')
    if not np.isfinite(p).all() or not np.isfinite(w).all() or np.any(p<0) or np.any(w<0):raise ValueError('Nonnegative finite observations required')
    minimum=w.min(axis=-1);mask=minimum>threshold*minimum.max(axis=(-2,-1),keepdims=True)
    q=np.zeros(p.shape,dtype=np.float32)
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        q[mask]=p[mask]/w[mask]
    np.nan_to_num(q,copy=False,nan=0.,posinf=np.finfo(np.float32).max,neginf=0.)
    np.clip(q,0,np.finfo(np.float32).max,out=q)
    return q,mask
