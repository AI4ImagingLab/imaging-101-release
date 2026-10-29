"""Explicit additive observation and nuclear/entrywise proximal operators.

Principal component pursuit, Candes et al (2011). MIT adaptation of
RobustPCA by Deepak Ganguli; complete original notice in LICENSE.
"""
import numpy as np


def forward(low_rank,sparse):
    """Normalized digital intensity D=L+S; no clipping or positivity constraint."""
    return low_rank+sparse


def shrink(matrix,threshold):
    """Exact elementwise proximal map for threshold*entrywise-L1."""
    return np.sign(matrix)*np.maximum(np.abs(matrix)-threshold,0)


def singular_threshold(matrix,threshold):
    """Exact nuclear proximal formula via the smaller symmetric Gram matrix.

    For X=U diag(sigma) V^T, return U diag(max(sigma-threshold,0)) V^T.
    In the tall case, eigenvectors V of X^T X yield X V diag(1-tau/sigma)
    V^T for retained sigma>tau; this avoids forming the unused large U.
    The wide branch uses X X^T analogously. No rank cap or altered objective.
    Floating differences from direct SVD are checked against independent
    original first and complete-pipeline outputs on the full native matrix.
    """
    if threshold<0:raise ValueError('Nonnegative nuclear threshold required')
    if matrix.shape[0]>=matrix.shape[1]:
        eigen,vec=np.linalg.eigh(matrix.T@matrix)
        sigma=np.sqrt(np.maximum(eigen,0));keep=sigma>threshold
        if not np.any(keep):return np.zeros_like(matrix)
        basis=vec[:,keep];return ((matrix@basis)*(1-threshold/sigma[keep]))@basis.T
    eigen,vec=np.linalg.eigh(matrix@matrix.T)
    sigma=np.sqrt(np.maximum(eigen,0));keep=sigma>threshold
    if not np.any(keep):return np.zeros_like(matrix)
    basis=vec[:,keep];return (basis*(1-threshold/sigma[keep]))@(basis.T@matrix)
