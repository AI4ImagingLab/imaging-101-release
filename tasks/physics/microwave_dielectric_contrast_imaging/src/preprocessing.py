from pathlib import Path
import numpy as np
def load_data(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k][0].copy() for k in f.files}
