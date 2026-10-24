from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load_data(path=None):
    with np.load(path or ROOT/'data/raw_data.npz',allow_pickle=False) as z:return {k:z[k][0].copy() for k in z.files}
