"""Explicit diagonally preconditioned conjugate gradients and Wiener map.

Algorithm conventions derived from MIT dachshund; see NOTICE.
"""
import time
import numpy as np
from .preprocessing import whiten,grid_coordinates,GRID_SHAPE
from .physics_model import (SIGNAL_VARIANCE,CORRELATION_LENGTHS,sparse_covariance,
                            whitened_product,map_from_coefficients)

def pcg_step(product,rhs,diagonal,x,r,d,delta,full_residual=False):
    """One original PCG update; return x,r,d,delta and q=A*d."""
    q=product(d)
    alpha=delta/np.dot(d,q)
    x=x+alpha*d
    r=rhs-product(x) if full_residual else r-alpha*q
    s=r/diagonal
    new_delta=np.dot(r,s)
    d=s+(new_delta/delta)*d
    return x,r,d,new_delta,q

def conjugate_gradients(product,rhs,diagonal,max_iter=500,tol=1e-3,step_r=50):
    """Zero initialization; source loop makes at most max_iter-1 updates."""
    if max_iter<2 or step_r<1 or tol<0:
        raise ValueError('Invalid PCG controls')
    x=np.zeros_like(rhs,dtype=np.float64)
    norm_b=np.linalg.norm(rhs)
    if norm_b==0:
        return x,{'success':False,'iterations':0,'residual_norms':np.array([0.0])}
    r=rhs-product(x)
    d=r/diagonal
    delta=np.dot(r,d)
    residuals=[np.linalg.norm(r)]
    success=False
    for iteration in range(1,max_iter):
        x,r,d,delta,_=pcg_step(product,rhs,diagonal,x,r,d,delta,iteration%step_r==0)
        residuals.append(np.linalg.norm(r))
        if residuals[-1]<=tol*norm_b:
            success=True
            break
    return x,{'success':success,'iterations':iteration,
              'residual_norms':np.asarray(residuals)}

def reconstruct(observations,tol=1e-3,max_iter=500,step_r=50,chunk_size=1024,
                lengths=CORRELATION_LENGTHS,variance=SIGNAL_VARIANCE,
                coordinates=None):
    """Return full field, data coefficients and convergence/timing diagnostics."""
    start=time.perf_counter()
    covariance=sparse_covariance(observations.positions,lengths=lengths,variance=variance)
    build_seconds=time.perf_counter()-start
    w,rhs=whiten(observations.noise,observations.flux)
    product=lambda v: whitened_product(covariance,w,v)
    y,info=conjugate_gradients(product,rhs,1+variance*w*w,max_iter,tol,step_r)
    alpha=w*y
    info.update({'covariance_seconds':build_seconds,'covariance_nnz':covariance.nnz,
                 'relative_system_residual':float(np.linalg.norm(rhs-product(y))/np.linalg.norm(rhs)),
                 'predicted_pixels':covariance@alpha})
    del covariance
    query=grid_coordinates() if coordinates is None else coordinates
    field=map_from_coefficients(observations.positions,alpha,query,chunk_size,lengths,variance)
    info['runtime_seconds']=time.perf_counter()-start
    if coordinates is None: field=field.reshape(GRID_SHAPE)
    return field,alpha,info
