"""Load observations without implicit histogram shifts or normalization."""
import numpy as np
def load_data(path):
    """Return uint16 (256,256,256) counts, uint64 (1,1,256) dye, bool mask."""
    with np.load(path,allow_pickle=False) as f:
        signal=f['measurements'][0].copy()
        calibration=f['calibration'][0].copy()
        mask=f['valid_mask'][0].astype(bool)
    if signal.shape!=(256,256,256) or calibration.shape!=(1,1,256) or mask.shape!=(256,256):
        raise ValueError('Unexpected observation shape')
    if signal.dtype!=np.uint16 or calibration.dtype!=np.uint64:
        raise ValueError('Unexpected photon-count dtype')
    return signal,calibration,mask
