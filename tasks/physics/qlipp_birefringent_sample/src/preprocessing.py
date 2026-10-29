"""Known camera dark subtraction, no target-dependent normalization."""
import numpy as np
BLACK_LEVEL_ADU=100.58

def load_data(path):
    """Load raw (1,position,channel,z,y,x), background and calibration."""
    with np.load(path,allow_pickle=False) as d:return {k:d[k] for k in d.files}

def subtract_dark(data,black_level=BLACK_LEVEL_ADU):
    """Return channel-first float32 raw/background ADU minus measured offset."""
    return (data['intensities'][0].transpose(1,0,2,3,4).astype(np.float32)-black_level,
            data['background'][0].astype(np.float32)-black_level)
