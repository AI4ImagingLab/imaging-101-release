"""Explicit two-chromophore closed-form inversion (modified Beer-Lambert)."""
import numpy as np
from .preprocessing import optical_density
from .physics_model import absorption_matrix

def invert_density(density, extinction, distances, ppf=6.0):
    """Return float64 HbO/HbR in M, (1,P,2,T), by explicit 2x2 cofactors.
    This is the full-rank two-wavelength case of the MNE pseudoinverse;
    no iteration, regularization, clipping, or temporal filtering is used.
    """
    a=absorption_matrix(extinction,distances,ppf)
    det=a[...,0,0]*a[...,1,1]-a[...,0,1]*a[...,1,0]
    if np.any(det == 0): raise ValueError('Singular extinction calibration')
    y=np.asarray(density,dtype=np.float64)
    out=np.empty_like(y)
    out[:,:,0,:]=(a[...,1,1,None]*y[:,:,0,:]-a[...,0,1,None]*y[:,:,1,:])/det[...,None]
    out[:,:,1,:]=(-a[...,1,0,None]*y[:,:,0,:]+a[...,0,0,None]*y[:,:,1,:])/det[...,None]
    return out

def reconstruct(intensity, extinction, distances, ppf=6.0):
    """Complete native recording -> relative HbO/HbR concentrations in M."""
    return invert_density(optical_density(intensity),extinction,distances,ppf)
