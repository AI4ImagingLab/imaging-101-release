"""Explicit Jacobi-preconditioned conjugate-gradient opacity inverse.

CG equations and stopping preserve the original documented configuration.
"""
import numpy as np
from .physics_model import matting_laplacian
from .preprocessing import partition_trimap


def conjugate_gradient(matrix, rhs, rtol=1e-7, maxiter=10000, capture_steps=()):
    x = np.zeros_like(rhs)
    residual = rhs-matrix.dot(x)
    norm_rhs = np.linalg.norm(rhs)
    norm_residual = np.linalg.norm(residual)
    trace = []; captured = {}
    if norm_residual < rtol*norm_rhs or norm_residual == 0:
        return x, np.empty((0, 3)), captured
    inverse_diagonal = 1/matrix.diagonal()
    z = residual*inverse_diagonal
    direction = z.copy()
    rz = np.inner(residual, z)
    for iteration in range(maxiter):
        action = matrix.dot(direction)
        alpha = rz/np.inner(direction, action)
        x += alpha*direction
        residual -= alpha*action
        norm_residual = np.linalg.norm(residual)
        trace.append([iteration+1, norm_residual, norm_residual/norm_rhs])
        if iteration+1 in capture_steps:
            captured[iteration+1] = x.copy()
        if norm_residual < rtol*norm_rhs:
            return x, np.asarray(trace), captured
        z = residual*inverse_diagonal
        beta = 1/rz
        rz = np.inner(residual, z)
        beta *= rz
        direction *= beta
        direction += z
    raise RuntimeError('original CG iteration budget exhausted')


def reconstruct(image, trimap, epsilon=1e-7, capture_steps=()):
    foreground, background, known, unknown = partition_trimap(trimap)
    laplacian = matting_laplacian(image, known.reshape(trimap.shape), epsilon)
    system = laplacian[unknown, :][:, unknown]
    rhs = -laplacian[unknown, :][:, known].dot(foreground[known])
    values, history, captured = conjugate_gradient(system, rhs, capture_steps=capture_steps)
    alpha = trimap.copy().ravel(); alpha[unknown] = values
    result = np.clip(alpha, 0, 1).reshape(trimap.shape)
    diagnostics = dict(history=history, captured=captured, laplacian=laplacian,
                       rhs=rhs, unclipped_unknown=values, known=known, unknown=unknown,
                       explicit_relative_residual=float(np.linalg.norm(system@values-rhs)/np.linalg.norm(rhs)))
    return result, diagnostics
