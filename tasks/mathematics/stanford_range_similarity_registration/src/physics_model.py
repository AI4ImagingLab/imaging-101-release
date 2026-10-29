"""Gaussian-mixture point acquisition and explicit E-step.

Adapted from MIT pycpd, Siavash Khallaghi; see LICENSE/NOTICE.
"""
import numpy as np

def transform(points,scale,rotation,translation):
    """Row-vector similarity: scale * points @ rotation + translation."""
    return scale*np.dot(points,rotation)+translation

def initialize_variance(target,source):
    """Published initial per-coordinate variance over every selected pair."""
    diff=target[None,:,:]-source[:,None,:]
    return np.sum(diff**2)/(target.shape[1]*len(target)*len(source))

def expectation(target,transformed_source,variance,outlier_fraction):
    """Return posterior P(source,target), marginal sums, weighted targets, mass.

    A uniform component competes with equal-weight isotropic Gaussian kernels.
    Coordinates are dimensionless; preserve upstream probability normalization.
    """
    m,d=transformed_source.shape;n=len(target)
    distance=np.sum((target[None,:,:]-transformed_source[:,None,:])**2,axis=2)
    p=np.exp(-distance/(2*variance))
    uniform=(2*np.pi*variance)**(d/2)*outlier_fraction/(1-outlier_fraction)*m/n
    denominator=np.sum(p,axis=0)
    denominator[denominator==0]=np.finfo(float).eps
    denominator+=uniform
    p=np.divide(p,np.tile(denominator,(m,1)))
    pt1=np.sum(p,axis=0);p1=np.sum(p,axis=1);mass=np.sum(p1)
    return p,p1,pt1,np.dot(p,target),mass
