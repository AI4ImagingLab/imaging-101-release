"""Positive quantum state fit with the published triangular parameterization.
Copyright 2020 University of Illinois Board of Trustees; MIT, see LICENSE.
"""
import numpy as np
from scipy.optimize import leastsq
from .physics_model import density,residual,predict
def initial_coordinates(counts):
    """Maximally mixed rho=I/4, upstream .0001 perturbation and count scale."""
    t=np.zeros(16);t[:4]=.5
    return (t+.0001)*np.sqrt(np.mean(counts)*4)
def reconstruct(counts,projectors):
    """Return rho(4,4), intensity scalar, diagnostic dict using all supplied rows.
    MINPACK LM ftol=xtol=1.49012e-8,gtol=0,maxfev=0 defaults.
    """
    t,_,info,message,status=leastsq(residual,initial_coordinates(counts),args=(counts,projectors),ftol=1.49012e-8,xtol=1.49012e-8,gtol=0.,maxfev=0,full_output=True)
    if status not in (1,2,3,4): raise RuntimeError(message)
    r=density(t,False); intensity=float(np.trace(r).real);r/=intensity
    return r,intensity,{'objective':float(np.sum(info['fvec']**2)),'evaluations':int(info['nfev']),'status':int(status)}
def linear_reconstruct(counts,projectors):
    """Unconstrained Hermitian least squares; intentionally does not clip eigenvalues."""
    basis=[]
    for i in range(4):
        a=np.zeros((4,4),complex);a[i,i]=1;basis.append(a)
    for i in range(4):
        for j in range(i+1,4):
            a=np.zeros((4,4),complex);a[i,j]=a[j,i]=1;basis.append(a)
            a=np.zeros((4,4),complex);a[i,j]=1j;a[j,i]=-1j;basis.append(a)
    design=np.array([predict(a,projectors) for a in basis]).T
    c=np.linalg.lstsq(design,counts,rcond=None)[0]
    r=np.einsum('i,ijk->jk',c,np.array(basis));scale=float(np.trace(r).real)
    return r/scale,scale
