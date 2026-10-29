"""Finite-b cumulant expansion of powder-averaged diffusion MRI."""
import numpy as np

def design_matrix(bvalues):
    """Columns [-b,b**2/6,1]; b in s/mm**2, D in mm**2/s."""
    b=np.asarray(bvalues,dtype=np.float64)
    return np.column_stack((-b,(1.0/6.0)*b**2,np.ones_like(b)))

def predict(diffusivity,kurtosis,baseline,bvalues):
    """Sbar(b)=S0 exp(-bD+b**2 D**2 K/6), ADU; no high-b extrapolation."""
    b=np.asarray(bvalues)
    return baseline[...,None]*np.exp(-diffusivity[...,None]*b+diffusivity[...,None]**2*kurtosis[...,None]*b*b/6)
