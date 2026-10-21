"""Window operations adapted from OpenPIV, copyright 2011 OpenPIV, GPL-3.0-or-later."""
from pathlib import Path
import numpy as np

def load_frames(path):
    """Return two float64 (369,511) arrays in digital counts."""
    with np.load(Path(path), allow_pickle=False) as d:
        frames = tuple(d[k][0].astype(np.float64) for k in ('frame_a', 'frame_b'))
    if any(f.shape != (369,511) or not np.isfinite(f).all() for f in frames):
        raise ValueError('Expected finite (369,511) frames')
    return frames

def windows(frame, size=64, overlap=16):
    """Return row-major stacked square windows, without edge padding."""
    step = size-overlap
    return np.stack([frame[y:y+size,x:x+size] for y in range(0,frame.shape[0]-size+1,step) for x in range(0,frame.shape[1]-size+1,step)])

def normalize(w):
    """Per-window population standard deviation, with zero for constants."""
    w = np.array(w,dtype=np.float64,copy=True)
    w -= w.mean(axis=(-2,-1),keepdims=True,dtype=np.float64)
    sd=w.std(axis=(-2,-1),keepdims=True)
    return np.divide(w,sd,out=np.zeros_like(w),where=sd!=0)

def prepare(a,b,window=32,search=64,overlap=16):
    """Normalize both windows and retain central interrogation support in A."""
    aa=normalize(windows(a,search,overlap));bb=normalize(windows(b,search,overlap))
    p=(search-window)//2
    mask=np.zeros((search,search));mask[p:search-p,p:search-p]=1
    return aa*mask,bb
