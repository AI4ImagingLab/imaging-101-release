"""Native detector values and observation-defined reference wavevectors."""
import numpy as np
NM_PER_PIXEL=.88
def load_data(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k][0] for k in f.files}
def prepare(image):
    a=image.astype(np.float64)
    return a-a.mean()
