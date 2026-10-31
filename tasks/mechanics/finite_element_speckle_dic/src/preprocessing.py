"""Load unnormalized, already oriented speckle observations."""
from pathlib import Path
import numpy as np


def load_data(path=Path('data/raw_data.npz')):
    """Return float64 reference/deformed images, nodes (N,2), connectivity (E,4)."""
    with np.load(path, allow_pickle=False) as data:
        return tuple(data[k][0].copy() for k in
                     ('reference_image', 'deformed_image', 'nodes', 'elements'))
