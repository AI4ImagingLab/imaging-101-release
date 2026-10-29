"""Observation loading preserves the repository-provided grayscale pixels."""
from pathlib import Path
import numpy as np

SHAPE=(1,4,1024,1024)

def load_observations(path: str | Path) -> np.ndarray:
    """Return (4,1024,1024) float64 holograms in digital-number units."""
    with np.load(path,allow_pickle=False) as data:
        h=data['holograms']
    if h.shape!=SHAPE or h.dtype!=np.uint8:
        raise ValueError(f'holograms must have shape {SHAPE} and dtype uint8')
    return h[0].astype(np.float64)

def center_frames(holograms: np.ndarray) -> np.ndarray:
    """Remove each frame's spatial mean, matching SOSR background convention."""
    h=np.asarray(holograms,dtype=np.float64)
    if h.ndim!=3 or h.shape[0]!=4 or not np.isfinite(h).all():
        raise ValueError('Expected four finite real holograms')
    return h-h.mean(axis=(-2,-1),keepdims=True)
