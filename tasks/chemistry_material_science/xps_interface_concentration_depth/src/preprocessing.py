"""Public acquisition data, with no hidden-profile access."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load_data(path=None):
    with np.load(path or ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        return {k:z[k][0].copy() for k in z.files}
def mean_observation(observations):
    """Mean of all 199 independent noisy acquisitions; units µm³."""
    return np.mean(observations,axis=0)
