"""Load unchanged author count observations in demagnified object coordinates."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load_observations(path=None):
    """Return x,y,fluence (200,200) float64, in cm, cm, and counts/pixel."""
    path = ROOT / 'data/raw_data.npz' if path is None else Path(path)
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k][0].copy() for k in ['x', 'y', 'fluence']}
    if any(a.shape != (200, 200) or a.dtype != np.float64 or not np.isfinite(a).all() for a in out.values()):
        raise ValueError('Expected finite float64 200 by 200 observations')
    if np.any(out['fluence'] <= 0):
        raise ValueError('The bound logarithmic model requires positive fluence')
    return out

def uniform_incident_fluence(fluence):
    """Equal-count uniform illumination; same float64 mean as original source."""
    return np.zeros_like(fluence) + np.mean(fluence)
