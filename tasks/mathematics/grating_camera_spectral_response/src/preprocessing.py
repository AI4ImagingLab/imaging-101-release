"""Measured crop reduction and independent population calibration, no target truth."""
from pathlib import Path
import numpy as np

CAMERAS=('eos','sony','iphone15promax','dji_pocket3')

def load_data(path=None):
    """Read public NPZ with batch axis removed; return independent ndarray copies."""
    path=Path(path) if path else Path(__file__).resolve().parents[1]/'data/raw_data.npz'
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k][0].copy() for k in z.files}

def observations(data,camera):
    """Return float64 direct RGB, dispersed LED RGB and unnormalized fluorescent RGB."""
    direct=np.median(data[camera+'_direct'].astype(np.float64).reshape(-1,3),axis=0)
    dispersed=np.median(data[camera+'_diffracted'].astype(np.float64),axis=0)
    fluorescent=np.median(data[camera+'_fluorescent'].astype(np.float64),axis=0)
    scale=direct.max()
    if not np.isfinite(scale) or scale<=0:raise ValueError('Nonpositive direct-light normalization')
    return direct/scale,dispersed/scale,fluorescent

def population_basis(population):
    """Return three31x31 uncentred eigenbases and mean31x3; population is N,31,3."""
    bases=[];means=[]
    for channel in range(3):
        matrix=population[:,:,channel].T.copy()
        matrix/=matrix.max(axis=0,keepdims=True)
        bases.append(np.linalg.eigh(matrix@matrix.T)[1][:,::-1])
        means.append(matrix.mean(axis=1))
    return np.asarray(bases),np.asarray(means).T
