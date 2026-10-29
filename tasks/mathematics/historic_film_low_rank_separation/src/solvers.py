"""Fixed-penalty PCP ADMM, local readable MIT mathematical extraction."""
import numpy as np
from .preprocessing import as_matrix,as_movie
from .physics_model import shrink,singular_threshold


def decompose(matrix,mu=None,lmbda=None,relative_tolerance=1e-7,max_iter=3000):
    """Return L,S,diagnostics; custom SVT/shrink/dual iterations are explicit.

    Matrix is float64 pixels-by-time, normalized digital intensity. Fixed mu
    matches source D.size/(4*sum(abs(D))). Lambda defaults1/sqrt(max(shape)).
    No centering, intensity clipping, randomized SVD or artificial iteration.
    """
    if not np.isfinite(matrix).all():raise ValueError('Finite observations required')
    norm=np.linalg.norm(matrix,'fro');total=np.sum(np.abs(matrix))
    if norm<=0 or total<=0:raise ValueError('Nonzero observations required')
    mu=matrix.size/(4*total) if mu is None else mu
    lmbda=1/np.sqrt(max(matrix.shape)) if lmbda is None else lmbda
    if mu<=0 or lmbda<=0 or relative_tolerance<=0:raise ValueError('Positive algorithm parameters required')
    inv=1/mu;sparse=np.zeros_like(matrix);dual=np.zeros_like(matrix);history=[]
    for i in range(max_iter):
        low=singular_threshold(matrix-sparse+inv*dual,inv)
        sparse=shrink(matrix-low+inv*dual,lmbda*inv)
        residual=matrix-low-sparse;dual=dual+mu*residual
        error=np.linalg.norm(residual,'fro')
        history.append([i+1,error,float(np.mean(np.abs(sparse)>1e-10))])
        if error<=relative_tolerance*norm:break
    if not history:raise ValueError('At least one iteration required')
    return low,sparse,{'history':np.asarray(history),'iterations':len(history),'relative_primal':float(error/norm),'converged':bool(error<=relative_tolerance*norm),'mu':float(mu),'lambda':float(lmbda)}


def reconstruct(frames,lambda_factor=1.,relative_tolerance=1e-7,max_iter=3000):
    """All native frames -> two signed full movie components and diagnostics."""
    matrix=as_matrix(frames)
    low,sparse,info=decompose(matrix,lmbda=lambda_factor/np.sqrt(max(matrix.shape)),relative_tolerance=relative_tolerance,max_iter=max_iter)
    return {'reconstruction':as_movie(low,frames.shape),'sparse_component':as_movie(sparse,frames.shape)},info
