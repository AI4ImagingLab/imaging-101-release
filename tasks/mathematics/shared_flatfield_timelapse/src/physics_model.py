"""Rank-one sensor-background model and public quality functional."""
import numpy as np
from scipy.fft import dctn

def background(flatfield,baseline):
    return np.asarray(baseline)[:,None,None]*np.asarray(flatfield)[None]

def quality(coarse,flatfield,baseline):
    flatfield=np.asarray(flatfield,dtype=np.float64)
    baseline=np.asarray(baseline,dtype=np.float64)
    mae=float(np.mean(abs(np.asarray(coarse,dtype=np.float64)-background(flatfield,baseline))))
    smooth=float(5/128**2*np.sum(abs(dctn(flatfield*baseline.mean(),norm="ortho"))))
    return dict(objective_counts=mae+smooth,mean_abs_residual_counts=mae,
                flatfield_mean_abs_error=float(abs(flatfield.mean()-1)))
