from pathlib import Path
import numpy as np
def load_data(path='data/raw_data.npz'):
    """Return unbatched copied real observations and instrument calibration."""
    with np.load(Path(path),allow_pickle=False) as f:return {k:f[k][0].copy() for k in f.files}
