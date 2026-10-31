"""Processed real TCSPC images; no truth or reconstruction input."""
from pathlib import Path
import numpy as np
def load_data(path='data/raw_data.npz'):
    with np.load(Path(path),allow_pickle=False) as z:return z['measurements'][0].copy(),z['calibration'][0].copy()
