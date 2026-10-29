"""Native CFIN signals and powder averaging."""
import numpy as np
B0_THRESHOLD_ADU=40.0
MIN_SIGNAL=0.0001

def load_data(path):
    """Signals (1,3,96,96,496), b values and gradient directions."""
    with np.load(path,allow_pickle=False) as d:return {k:d[k] for k in d.files}

def signal_mask(signals,bvalues,threshold=B0_THRESHOLD_ADU):
    """Observation-only b=0 mean >40 ADU; no fitted-parameter screening."""
    return signals[...,bvalues==0].mean(axis=-1)>threshold

def shell_average(signals,bvalues):
    """Return means, direction counts and sorted exact deposited shells."""
    shells=np.unique(bvalues)
    means=np.stack([signals[...,bvalues==b].mean(axis=-1) for b in shells],axis=-1)
    counts=np.array([(bvalues==b).sum() for b in shells],dtype=np.float64)
    return means,counts,shells
