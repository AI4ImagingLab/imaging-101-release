from pathlib import Path
import numpy as np

def load(path="data/raw_data.npz"):
 with np.load(path,allow_pickle=False) as f:return {k:f[k][0] for k in f.files}
