"""Explicit zero-normal-flux finite differences and conditional target operator."""
import numpy as np
def gradient(image):
    """Forward column/row differences; zero final column/row, extra color axes kept."""
    a=np.asarray(image,dtype=np.float64);g=np.zeros(a.shape+(2,),dtype=np.float64)
    g[:,:-1,...,0]=a[:,1:]-a[:,:-1];g[:-1,...,1]=a[1:]-a[:-1]
    return g
def divergence(flux):
    """Source backward divergence on fluxes with zero outgoing final boundaries.

    The public reconstruction always supplies gradients with those zeros;
    arbitrary external fluxes must satisfy the same boundary convention.
    """
    q=np.asarray(flux,dtype=np.float64);a=q[...,0].copy()+q[...,1]
    a[:,1:]-=q[:,:-1,...,0];a[1:]-=q[:-1,...,1]
    return a
def laplacian(image):
    return divergence(gradient(image))
def threshold_gradients(image,h=.033):
    """Per-channel gradient magnitude threshold; strict >h source API convention."""
    if not np.isfinite(h) or h<0:raise ValueError('h must be finite and nonnegative')
    g=gradient(image);keep=np.sqrt(g[...,0]**2+g[...,1]**2)>h
    return g*keep[...,None]
def right_hand_side(image,h=.033,epsilon=1e-8):
    return laplacian(divergence(threshold_gradients(image,h)))+epsilon*np.asarray(image)
def stationarity(image,reconstruction,h=.033,epsilon=1e-8):
    rhs=right_hand_side(image,h,epsilon)
    residual=laplacian(laplacian(reconstruction))+epsilon*reconstruction-rhs
    return float(np.linalg.norm(residual)/(np.linalg.norm(rhs) or 1.))
