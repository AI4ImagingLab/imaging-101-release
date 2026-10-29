"""Explicit Gaussian mixture and conditional-autoregressive mathematics.
CARD contributors, GPL-3.0; see test/validation.md for source mapping.
"""
import numpy as np


def forward_model(basis,weights):
    """Predicted source-scaled gene expression, genes by spatial locations."""
    return np.asarray(basis)@np.asarray(weights).T


def car_precision(kernel,phi):
    """Symmetric D-phi W, D=diag(row sums W), 0<=phi<1."""
    if not 0<=phi<1:raise ValueError("CAR phi must lie in [0,1)")
    return np.diag(kernel.sum(axis=1))-phi*kernel


def log_objective(X,basis,V,b,variance,lambdas,phi,kernel):
    """Direct-residual Gaussian/CAR/inverse-gamma score, up to fixed-phi normalizer."""
    residual=np.asarray(X)-forward_model(basis,V)
    L=car_precision(kernel,phi);Z=V-np.asarray(b)[None,:]
    quadratic=np.sum(Z*(L@Z),axis=0)
    n,k=V.shape
    return float(-X.size/2*np.log(variance)-np.sum(residual**2)/(2*variance)
                 -n/2*np.log(lambdas).sum()-.5*np.sum(quadratic/lambdas)
                 -2*np.log(lambdas).sum()-np.sum((n/2)/lambdas))


def normalize_weights(V):
    totals=V.sum(axis=1,keepdims=True)
    if np.any(totals<=0):raise ValueError("Cannot normalize zero mixture weight")
    return V/totals


def car_log_normalizer(kernel,phi,n_types):
    """SI Eq9: +K/2 logdet(L), essential when comparing different phi."""
    sign,logdet=np.linalg.slogdet(car_precision(kernel,phi))
    if sign<=0:raise ValueError("CAR precision must be positive definite")
    return float(n_types*.5*logdet)
