"""Closed-form lifetime inversion: no optimization or learned weights."""
import numpy as np
from .preprocessing import load_data
from .physics_model import normalized_phasor,calibrate,lifetimes
def reconstruct(path):
    """Read raw NPZ and return batch-first float64 lifetime maps (ns)."""
    counts,dye,mask=load_data(path)
    _,g,s=normalized_phasor(counts)
    g,s=calibrate(g,s,dye)
    phase,mod=lifetimes(g,s)
    return {'reconstruction':np.where(mask,phase,0)[None],
            'modulation_lifetime':np.where(mask,mod,0)[None]}
