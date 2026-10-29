"""Strict loading of normalized, calibrated pressure-gradient observations."""
from pathlib import Path
import numpy as np
PRESSURE_SCALE_PA = 0.001
LENGTH_SCALE_M = 0.001

def load_data(path="data/raw_data.npz"):
    """Return gradient (2,H,W), coordinates (W,), both float32."""
    with np.load(Path(path), allow_pickle=False) as z:
        g = z["pressure_gradient"]
        q = z["coordinate"]
    if g.dtype != np.float32 or q.dtype != np.float32 or g.ndim != 4 or g.shape[:2] != (1, 2):
        raise ValueError("Invalid batch-first gradient or coordinate dtype/shape")
    if q.shape != (1, g.shape[-1]) or g.shape[-2] != g.shape[-1] or not np.isfinite(g).all() or not np.isfinite(q).all():
        raise ValueError("Inconsistent finite square-grid observations")
    if not np.all(np.diff(q[0]) > 0):
        raise ValueError("Coordinates must increase")
    return g[0], q[0]
