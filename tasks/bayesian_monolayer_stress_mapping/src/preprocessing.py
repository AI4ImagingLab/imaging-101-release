"""Author-existing observations; no new synthetic data are generated."""
from pathlib import Path
import numpy as np
ROWS=50
COLUMNS=50
SPACING=2.0
def load_data(path="data/raw_data.npz"):
    """Return copied traction (2,50,50), float64 kPa, axes component,y,x."""
    with np.load(Path(path),allow_pickle=False) as data:
        value=data["traction"]
        if value.shape!=(1,2,ROWS,COLUMNS) or value.dtype!=np.dtype("float64") or not np.isfinite(value).all():raise ValueError("Expected finite float64 traction (1,2,50,50)")
        return value[0].copy()
def coordinates():
    """Return x,y cell-center coordinates in micrometres."""
    return np.arange(1.,100.,2.),np.arange(1.,100.,2.)
