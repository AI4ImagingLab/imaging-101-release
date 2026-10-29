"""Explicit complex Fourier acquisition, adjoint and sampling response.

Adapted mathematics from RM-Tools, copyright (c) 2020 CIRADA, MIT; see ../LICENSE.
"""
import numpy as np

def forward(components,lambda2,phi,lambda0):
    """Channel polarization for complex per-bin flux, exp(+2i phi (lambda²-lambda0²))."""
    return np.exp(2j*np.outer(lambda2-lambda0,phi))@components

def adjoint(polarization,lambda2,phi,lambda0):
    """Unnormalized Hermitian adjoint of forward, no weights or bin spacing."""
    return np.exp(-2j*np.outer(phi,lambda2-lambda0))@polarization

def dirty_spectrum(polarization,lambda2,phi,weights,lambda0):
    """Weighted inverse Fourier sum divided by sum(weights); units Jy/beam."""
    return adjoint(polarization*weights,lambda2,phi,lambda0)/np.sum(weights)

def rotation_measure_response(lambda2,phi,weights,lambda0):
    """Double-extent RMSF and selected source analytical width 3.8 / delta(lambda²)."""
    n=len(phi);extension=int(np.ceil(n/2.))
    phi2=phi[0]+(np.arange(n+2*extension)-extension)*(phi[1]-phi[0])
    response=np.sum(weights*np.exp(np.outer(-2j*phi2,lambda2-lambda0)),axis=1)/np.sum(weights)
    width=3.8/(np.max(lambda2)-np.min(lambda2))
    return response,phi2,float(width)
