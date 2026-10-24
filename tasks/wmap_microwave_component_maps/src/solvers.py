"""Explicit fixed-spectrum component separation adapted from FGBuster, GPL-3.0."""
import numpy as np
from .physics_model import mixing_matrix,forward
def separate(measurements,mixing):
    """SVD maximum-likelihood amplitudes for equal channel weights.
    Return maps(C,Y,X), response covariance(C,C), weights(C,F).
    No regularizer or iteration is needed for this closed-form inverse.
    """
    u,s,vh=np.linalg.svd(mixing,full_matrices=False)
    weights=(vh.T/s)@u.T
    maps=np.einsum('cf,fyx->cyx',weights,measurements)
    covariance=(vh.T/(s*s))@vh
    return maps,covariance,weights
def reconstruct(data,beta_sync=-3.,beta_dust=1.54,dust_temperature=20.):
    """Return separated maps(3,48,64), model prediction(5,48,64), covariance(3,3)."""
    A=mixing_matrix(data['frequencies'],beta_sync,beta_dust,dust_temperature)
    maps,covariance,_=separate(data['measurements'],A)
    return maps,forward(maps,A),covariance
