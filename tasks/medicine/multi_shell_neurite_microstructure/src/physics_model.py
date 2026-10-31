"""Explicit two-compartment spherical-mean acquisition model.

Adapted from SMT, Copyright2016 Enrico Kaden & University College London,
BSD-2-Clause; full notice is retained in NOTICE. No domain solver is imported.
"""
import numpy as np
from scipy.special import erf

DIFFUSIVITY_MAX=3.05e-3

def spherical_mean(bvalues,parallel,perpendicular):
    """Uniform orientation integral; b in s/mm² and diffusivities in mm²/s."""
    b,p,t=np.broadcast_arrays(np.asarray(bvalues,float),np.asarray(parallel,float),np.asarray(perpendicular,float))
    high=np.maximum(p,t);low=np.minimum(p,t);delta=b*(high-low)
    root=np.sqrt(np.maximum(delta,0));ratio=np.ones_like(root)
    np.divide(np.sqrt(np.pi)*erf(root),2*root,out=ratio,where=root!=0)
    return np.exp(-b*low)*ratio

def transform_parameters(logits):
    """Original bounded exp(x)/(1+exp(x)), with its overflow guard."""
    z=np.asarray(logits,float);upper=np.array([1.,DIFFUSIVITY_MAX]);cap=np.log(np.finfo(float).max)
    exponential=np.exp(np.minimum(z,cap));value=upper*exponential/(1+exponential)
    return np.where(z>cap,upper,value)

def predicted_signal(parameters,bvalues,b0):
    """Return powder-mean signal at each supplied b value, preserving order."""
    p=np.asarray(parameters,float);b=np.asarray(bvalues,float)
    v=p[:,0,None];d=p[:,1,None]
    return np.asarray(b0)[:,None]*(v*spherical_mean(b[None,:],d,0)+(1-v)*spherical_mean(b[None,:],d,(1-v)*d))

def signal_objective(logits,observed,bvalues):
    """Original unnormalized SSE on every positive-b directional observation.

    Repeated b values share the same explicit model evaluation; every measured
    directional residual is retained, rather than silently changing weights.
    """
    y=np.asarray(observed,float);b=np.asarray(bvalues,float).ravel();positive=b>0
    unique,inverse=np.unique(b[positive],return_inverse=True)
    b0=y[:,b==0].mean(1)
    predicted=predicted_signal(transform_parameters(logits),unique,b0)[:,inverse]
    # Advanced-index gathering can produce Fortran order for n>1 but both
    # C/F order for n=1. Fix the row reduction layout so batch partitioning
    # cannot change the summation tree and hence a tightly stopped simplex.
    squared=np.ascontiguousarray((y[:,positive]-predicted)**2)
    return np.sum(squared,axis=1)
