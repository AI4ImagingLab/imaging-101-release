"""Exponential spin echo model; Fabien Georget (2020), BSD-3-Clause."""
import numpy as np
RELAXATION_TIME=np.logspace(-5,0,255)
def decay_kernel(echo_time,relaxation_time=RELAXATION_TIME):
    """(E,) seconds and (T,) seconds -> (E,T) dimensionless decay operator."""
    return np.exp(-np.asarray(echo_time)[:,None]/np.asarray(relaxation_time)[None,:])
def objective(amplitude,signal,kernel,alpha=1e-4):
    r=kernel@amplitude-signal
    return float(r@r+alpha*(amplitude@amplitude))
