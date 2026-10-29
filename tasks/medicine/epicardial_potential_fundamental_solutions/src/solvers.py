"""Explicit thin-SVD Tikhonov and complete CRESO parameter search."""
import numpy as np
from .physics_model import virtual_sources,fundamental_matrix,evaluate_potential
def ridge_coefficients(u,s,vt,rhs,lamb):
    filtered=s/(s*s+lamb*lamb)
    return (vt.T@np.diag(filtered)@u.T)@rhs
def creso_curve(s,projection,lambdas):
    # Each candidate evaluates d/d(lambda) of the CRESO residual/smoothing
    # combination, using the source variant's squared regularization value.
    sigma=s[None,:];alpha=projection[None,:];lamb=np.asarray(lambdas)[:,None]
    return np.sum(sigma**2*alpha**2*(sigma**2-3*lamb**2)/(sigma**2+lamb**2)**3,axis=1)
def reconstruct(data):
    sources,normals=virtual_sources(data);indices=data['electrode_indices'].ravel()
    points=data['torso_vertices'][indices];measurement=data['potentials'].ravel()
    matrix=fundamental_matrix(points,sources,normals[indices]);rhs=np.r_[measurement,np.zeros(len(measurement))]
    u,s,vt=np.linalg.svd(matrix,full_matrices=False)
    lambdas=np.linspace(1e-8,.001,10000);curve=creso_curve(s,u.T@rhs,lambdas)
    selected=float(lambdas[np.argmax(curve)])
    coef=ridge_coefficients(u,s,vt,rhs,selected)
    epi=evaluate_potential(data['sock_vertices'],sources,coef)
    return {'reconstruction':epi[None],'selected_lambda':selected,'creso':curve,'lambdas':lambdas,
            'coefficients':coef,'sources':sources,'operator':matrix,'rhs':rhs,'singular_values':s,
            'predicted_cauchy':matrix@coef}
