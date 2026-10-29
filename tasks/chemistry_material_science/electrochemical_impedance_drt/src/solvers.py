"""Nonnegative Tikhonov DRT; no electrochemical reconstruction library."""
import numpy as np
from scipy.optimize import nnls
from .physics_model import operators
REGULARIZATION=1e-3
def solve(frequency,impedance,regularization=REGULARIZATION):
    a,l,tau=operators(frequency)
    matrix=np.vstack([a.real,a.imag,np.sqrt(regularization)*l]);rhs=np.r_[impedance.real,impedance.imag,np.zeros(len(l))]
    x,_=nnls(matrix,rhs,maxiter=10000)
    return x,a@x
def reconstruct(data):
    x,fit=solve(data['frequency'],data['impedance'])
    return {'reconstruction':x[None,2:],'series_parameters':x[None,:2],'fitted_impedance':fit[None,:]}
