from pathlib import Path
import numpy as np
def load_data(path='data/raw_data.npz'):
    """Read the real five-band observation patch; remove only the batch axis."""
    with np.load(Path(path),allow_pickle=False) as f:return {k:f[k][0].copy() for k in f.files}
