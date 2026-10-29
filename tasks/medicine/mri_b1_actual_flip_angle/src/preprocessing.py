"""Lossless native-volume observation loading."""
import numpy as np
def load_data(path):
    """Return float32 (88,128,32,2) observations and bool (88,128,32) mask."""
    with np.load(path,allow_pickle=False) as f:
        signal=f['measurements'][0].copy();mask=f['valid_mask'][0].astype(bool)
    if signal.shape!=(88,128,32,2) or mask.shape!=(88,128,32):raise ValueError('Unexpected native volume shape')
    if signal.dtype!=np.float32:raise ValueError('Expected original float32 magnitude observations')
    if not np.all(np.isfinite(signal)) or np.any(signal[mask]<=0):raise ValueError('Invalid magnitude signal')
    return signal,mask
