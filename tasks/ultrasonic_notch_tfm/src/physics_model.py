"""Homogeneous longitudinal propagation and reciprocal HMC weighting."""
import numpy as np
def reconstruction_grid():
    return np.linspace(-.02,.02,201),np.linspace(0.,.05,251)
def travel_times(positions,x,z,speed=6300.):
    xx,zz=np.meshgrid(x,z,indexing='xy')
    return np.sqrt((xx.ravel()[:,None]-positions[None,:,0])**2+positions[None,:,1]**2+(zz.ravel()[:,None]-positions[None,:,2])**2)/speed
def reciprocity_weights(tx,rx):
    return np.where(tx==rx,1.,2.)
