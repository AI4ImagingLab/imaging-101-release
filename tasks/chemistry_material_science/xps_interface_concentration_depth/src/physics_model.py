"""Explicit calibrated cylindrical photoelectron acquisition and adjoint."""
import numpy as np
def forward(profile,system_matrix):
    """(201,) relative density -> (10,) geometrically normalized area, µm³."""
    return system_matrix@profile
def adjoint(values,system_matrix):
    return system_matrix.T@values
def second_difference(n):
    """Author unscaled [-1,2,-1] interior stencil, no fixed endpoints."""
    a=np.zeros((n-2,n),dtype=np.float64)
    i=np.arange(n-2);a[i,i]=-1;a[i,i+1]=2;a[i,i+2]=-1
    return a
def objective(profile,matrix,mean,variance,prior_variance):
    d=matrix@profile-mean;r=second_difference(profile.size)@profile
    return float(d@d/variance+r@r/prior_variance)
