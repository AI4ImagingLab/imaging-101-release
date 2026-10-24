"""Explicit Simpson-smoothed arterial convolution, units and adjoint.

Adapted from MIT DSC toolbox; see ../LICENSE and private provenance.
"""
import numpy as np
from scipy.linalg import toeplitz

def convolution_matrix(arterial_input,circulant=True):
    """Return unscaled80x80 causal or160x160 circular acquisition matrix.

    Input s^-1; multiplying by the1.55s sampling interval and a flow-scaled
    residue in s^-1 yields the observed relaxation-rate proxy in s^-1.
    Simpson boundary weights are exactly those of the selected source.
    """
    a=np.asarray(arterial_input,dtype=np.float64)
    if a.ndim!=1 or len(a)<3 or not np.isfinite(a).all():raise ValueError('Finite1-D arterial input required')
    n=len(a)
    if circulant:
        col=np.zeros(2*n);col[0]=a[0]
        col[1:n-1]=(a[:-2]+4*a[1:-1]+a[2:])/6
        col[n-1]=(a[-2]+4*a[-1])/6;col[n]=a[-1]/6
        return toeplitz(col,np.r_[col[0],col[:0:-1]])
    col=a.copy();col[1:-1]=(a[:-2]+4*a[1:-1]+a[2:])/6
    return toeplitz(col,np.r_[col[0],np.zeros(n-1)])

def forward(residue,arterial_input,tr=1.55,circulant=True):
    """Return the acquisition window; preserve signed residue and physical scale."""
    matrix=convolution_matrix(arterial_input,circulant)
    return (np.asarray(residue)@(tr*matrix).T)[...,:len(arterial_input)]

def adjoint(observations,arterial_input,tr=1.55,circulant=True):
    """Euclidean transpose of the complete declared acquisition operator."""
    matrix=convolution_matrix(arterial_input,circulant)
    return np.asarray(observations)@(tr*matrix[:len(arterial_input)])
