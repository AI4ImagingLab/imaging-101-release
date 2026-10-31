"""Read unchanged author SAC samples packaged as NPZ; no domain dependencies."""
from pathlib import Path
import numpy as np
SHIFTS = np.array([32,32,31,31,31,31,31,32])
NPTS = 150
DEPTHS = np.array([10.,12.,20.])
STATIONS = ('BK.QRDG.00','BK.RUSS.00','BK.CVS.00','BK.OAKV.00','BK.FARB.00','BK.SAO.00','BK.CMB.00','BK.MNRC.00')

def load_data(path='data/raw_data.npz'):
    """Return unbatched arrays, with native float32 samples and float64 geometry."""
    with np.load(Path(path), allow_pickle=False) as f:
        return {k:f[k][0].copy() for k in f.files}

def window_waveforms(waveforms):
    """(8,3,231) cm -> (8,3,150) float64 cm; native station windows."""
    return np.stack([waveforms[i,:,s:s+NPTS] for i,s in enumerate(SHIFTS)]).astype(np.float64)

def distance_weights(distance):
    """Native objective weights distance/min(distance), not squared distances."""
    return np.asarray(distance,dtype=np.float64)/np.min(distance)
