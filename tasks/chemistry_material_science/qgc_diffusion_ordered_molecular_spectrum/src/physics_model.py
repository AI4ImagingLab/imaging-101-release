"""Pulsed-gradient attenuation in a logarithmic diffusion amplitude basis."""
import numpy as np
DIFFUSION=np.geomspace(.1,50,256)*1e-10

def kernel(b_values,diffusion=DIFFUSION):
    # Preserve author's dimensionless scaling and multiplication convention.
    return np.exp(-(np.asarray(b_values)/1e10)[:,None]*(np.asarray(diffusion)/1e-10)[None,:])

def forward(amplitudes,b_values):
    return np.asarray(amplitudes)@kernel(b_values).T
