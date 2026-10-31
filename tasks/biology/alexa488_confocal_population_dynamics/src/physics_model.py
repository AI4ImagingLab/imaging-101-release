# Adapted mathematics from PyCorrFit (GPL-2.0-only), Paul Mueller and contributors.
import numpy as np
TRIPLET_MS = 0.002
ASPECT = 5.0

def correlation_model(parameters, lag_ms):
    """[N,T,tauD_ms] -> excess correlation at lag_ms, dimensionless."""
    n, triplet, residence = np.asarray(parameters,dtype=float)
    t=np.asarray(lag_ms,dtype=float)
    diffusion=1/((1+t/residence)*np.sqrt(1+t/(ASPECT**2*residence)))
    blinking=1+triplet/(1-triplet)*np.exp(-t/TRIPLET_MS)
    return diffusion*blinking/n

def objective(parameters, lag_ms, observations):
    r=correlation_model(parameters,lag_ms)-observations
    return float(r@r)
