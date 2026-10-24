"""Lossless real-HDR input loading; native lower clipping only."""
from pathlib import Path
import numpy as np

HEIGHT=2048
WIDTH=2448
FRAMES=16
ROOT=Path(__file__).resolve().parents[1]

def load_observations(path=None):
    path=ROOT/'data/raw_data.npz' if path is None else Path(path)
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'intensity','polarizer_psg','polarizer_psa'}:raise ValueError('Exactly the three public keys required')
        data={k:z[k].copy() for k in z.files}
    for k,a in data.items():
        shape=(1,16,HEIGHT,WIDTH) if k=='intensity' else (1,16)
        if a.shape!=shape or a.dtype.kind!='f' or not np.isfinite(a).all():raise ValueError('Invalid public '+k)
    return {k:a[0] for k,a in data.items()}

def prepare_intensity(intensity):
    """Preserve HDR values above1; author's numerical lower bound is zero."""
    a=np.asarray(intensity)
    if a.ndim!=3 or not np.isfinite(a).all():raise ValueError('Finite(frame,row,column) intensities required')
    return np.clip(a,0,None)
