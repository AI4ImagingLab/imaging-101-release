"""Author-native hydraulic survey geometry; no target-derived calibration."""
import numpy as np
NROW=75
NCOL=100
GRID_METRES=10.
WELL_RATE=25.
RECHARGE=.001
NOISE_VARIANCE=.25

def prepare(raw):
    heads=np.asarray(raw['hydraulic_head']);wells=np.asarray(raw['well_indices'])
    if heads.shape!=(1,30,29) or heads.dtype.kind!='f' or not np.isfinite(heads).all():raise ValueError('Expected finite floating heads(1,30,29)')
    if wells.shape!=(1,30,2) or wells.dtype.kind not in 'iu':raise ValueError('Expected integer well_indices(1,30,2)')
    if np.any(wells[0,:,0]<0) or np.any(wells[0,:,0]>=NROW) or np.any(wells[0,:,1]<=0) or np.any(wells[0,:,1]>=NCOL-1):raise ValueError('Wells must lie in active aquifer cells')
    if len(np.unique(wells[0],axis=0))!=30:raise ValueError('Duplicate survey wells')
    return heads[0].reshape(-1).astype(np.float64),wells[0].astype(np.int64)
