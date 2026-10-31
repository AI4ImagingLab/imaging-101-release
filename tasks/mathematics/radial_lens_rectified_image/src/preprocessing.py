"""Observed image/line data and original radius normalization."""
from pathlib import Path
import numpy as np
CENTER=np.array([344.,253.])
def load_data(path=None):
    path=Path(path) if path else Path(__file__).resolve().parents[1]/'data/raw_data.npz'
    with np.load(path,allow_pickle=False) as f:return {k:f[k][0] for k in f.files}
def normalize_points(points):
    p=np.asarray(points,dtype=float)-CENTER
    factor=np.sqrt(sum(float(x*x+y*y) for x,y in p)/len(p))
    return p/factor,factor
