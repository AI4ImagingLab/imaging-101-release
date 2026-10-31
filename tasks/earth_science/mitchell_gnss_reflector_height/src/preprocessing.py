from pathlib import Path
import numpy as np
def load_data(path=None):
    path=Path(path) if path else Path(__file__).resolve().parents[1]/'data/raw_data.npz'
    with np.load(path,allow_pickle=False) as z:return {k:z[k][0].copy() for k in z.files}
