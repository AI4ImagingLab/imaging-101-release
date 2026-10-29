from pathlib import Path
import numpy as np
def load_data(path='data/raw_data.npz'):
    """Load calibrated observations and geometry; remove the single batch axis."""
    with np.load(Path(path),allow_pickle=False) as a:
        return {k:a[k][0].copy() for k in a.files}
