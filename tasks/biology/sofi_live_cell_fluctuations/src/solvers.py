"""Zero-lag moment-SOFI, second and fourth central-moment images."""
import numpy as np
from .preprocessing import temporal_mean
from .physics_model import moments

def reconstruct(intensities,frames=None):
    """Return float64 (1,block,y,x) cumulant2 [ADU**2] and reconstruction moment4 [ADU**4].

    frames optionally uses first N frames per block for convergence experiments.
    Default processes all observations. No clipping, normalization or deconvolution.
    """
    c2=[];m4=[]
    for block in intensities[0]:
        stack=block if frames is None else block[:frames]
        ms=moments(stack,temporal_mean(stack),4)
        c2.append(ms[2].astype(np.float64));m4.append(ms[4].astype(np.float64))
    return {'cumulant2':np.stack(c2)[None],'reconstruction':np.stack(m4)[None]}
