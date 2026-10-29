"""Observation interface; all lengths are Angstrom, no offset correction."""
from pathlib import Path
import numpy as np
PIXEL_PITCH = 15.0
NOISE_SIGMA = 3.0
SURFACE_SHAPE = (1, 30, 30, 30)
def load_observations(path="data/raw_data.npz"):
    """Return (measurements[30,30,30], probe[14,14]), both float64 in A."""
    with np.load(Path(path), allow_pickle=False) as z:
        image, probe = z["measurements"], z["probe"]
    if image.shape != SURFACE_SHAPE or probe.shape != (1,14,14):
        raise ValueError("Invalid observation or probe shape")
    if image.dtype != np.float64 or probe.dtype != np.float64:
        raise ValueError("Observations must preserve stored float64")
    if not np.isfinite(image).all() or not np.isfinite(probe).all():
        raise ValueError("Nonfinite observations")
    return image[0], probe[0]
