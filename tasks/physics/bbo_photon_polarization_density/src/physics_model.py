"""Born acquisition and positive-matrix coordinates, extracted from KwiatLab.
Copyright 2020 University of Illinois Board of Trustees; MIT, see LICENSE.
"""
import numpy as np
def triangular(t):
    """Map 16 real coordinates to a 4x4 lower triangular complex matrix.
    Ordering: real main diagonal, real/imaginary first subdiagonal,
    real/imaginary second subdiagonal, real/imaginary third subdiagonal.
    """
    t=np.asarray(t); out=np.zeros((4,4),complex); k=0
    for d in range(4):
        n=4-d
        out+=np.diag(t[k:k+n],-d); k+=n
        if d: out+=1j*np.diag(t[k:k+n],-d); k+=n
    return out
def density(t,normalize=True):
    """Return T T-dagger, optionally divided by its real trace."""
    a=triangular(t); r=a@a.conj().T
    return r/np.trace(r) if normalize else r
def predict(rho,projectors,intensity=1.):
    """Born expected corrected counts: intensity*real(trace(P_j rho))."""
    return np.array([intensity*np.real(np.trace(p@rho)) for p in projectors])
def residual(t,counts,projectors):
    """Upstream Pearson-like least-squares residual, expected-count floor .01."""
    expected=np.maximum(predict(density(t,False),projectors),.01)
    return (expected-counts)/np.sqrt(expected)
