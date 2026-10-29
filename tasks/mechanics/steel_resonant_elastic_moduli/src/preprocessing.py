"""Public observed peak frequencies and author analysis preferences only."""
from pathlib import Path
import numpy as np

def load_data(path=Path('data/raw_data.npz')):
    with np.load(path,allow_pickle=False) as f:
        return f['frequencies'][0],f['analysis_weights'][0]
