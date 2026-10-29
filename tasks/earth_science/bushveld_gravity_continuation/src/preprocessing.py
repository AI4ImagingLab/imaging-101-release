"""Public acquisition arrays; no fitted or reference information is loaded."""
from pathlib import Path
import numpy as np

def load_observations(path):
    """Return station (N,3) metres, data (N,) mGal, image grid (H,W,3) metres."""
    with np.load(Path(path), allow_pickle=False) as a:
        coordinates=a['station_coordinates'][0].copy()
        data=a['gravity_disturbance'][0].copy()
        grid=a['image_coordinates'][0].copy()
    if coordinates.shape != (306,3) or data.shape != (306,) or grid.shape != (48,64,3):
        raise ValueError('Unexpected acquisition shape')
    if not all(np.isfinite(x).all() for x in [coordinates,data,grid]):
        raise ValueError('Nonfinite acquisition')
    return coordinates,data,grid
