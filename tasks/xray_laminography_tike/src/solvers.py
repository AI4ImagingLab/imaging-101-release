"""Iterative solvers for laminographic/tomographic reconstruction.

Implements conjugate gradient descent for the least-squares
laminography problem using the Dai-Yuan conjugate direction formula
and backtracking line search.

No dependency on tike. Uses cupy for GPU acceleration.
"""

import logging
import warnings

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from src.physics_model import (
    forward_project,
    adjoint_project,
    cost_function,
    gradient,
)

logger = logging.getLogger(__name__)


def _get_xp(arr):
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def _to_gpu(arr):
    if cp is not None:
        return cp.asarray(arr)
    return arr


def _to_cpu(arr):
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _estimate_step_length(obj, theta, tilt):
    """Estimate initial CG step length from forward-adjoint scaling.

    Computes ||adjoint(forward(obj))|| / ||obj|| and multiplies by 2
    (over-estimate for safety).
    """
    xp = _get_xp(obj)
    n = obj.shape[0]
    fwd = forward_project(obj, theta, tilt)
    adj = adjoint_project(fwd, theta, tilt, n)
    norm_adj = float(xp.sqrt(xp.sum(xp.abs(adj) ** 2).real))
    norm_obj = float(xp.sqrt(xp.sum(xp.abs(obj) ** 2).real)) + 1e-32
    scaler = norm_adj / norm_obj
    return 2.0 * scaler if np.isfinite(scaler) else 1.0


def _direction_dai_yuan(xp, grad1, grad0=None, dir_prev=None):
    """Compute search direction using the Dai-Yuan formula.

    beta = ||grad1||^2 / <dir_prev, grad1 - grad0>

    On first iteration (no previous direction), returns -grad1.
    """
    if dir_prev is None:
        return -grad1

    diff = grad1 - grad0
    denom = float(xp.sum(xp.conj(dir_prev) * diff).real) + 1e-32
    beta = float(xp.sum(xp.abs(grad1) ** 2).real) / denom
    return -grad1 + beta * dir_prev


def _line_search(cost_fn, x, d, step_length, cost_x=None, step_shrink=0.5):
    """Backtracking line search.

    Reduces step_length until cost(x + step * d) <= cost(x).
    """
    xp = _get_xp(x)
    if cost_x is None:
        cost_x = cost_fn(x)

    step_count = 0
    first_step = step_length
    while True:
        x_new = x + step_length * d
        cost_new = cost_fn(x_new)
        if cost_new <= cost_x:
            break
        step_length *= step_shrink
        if step_length < 1e-32:
            warnings.warn("Line search failed for conjugate gradient.")
            return 0.0, cost_x, x
        step_count += 1

    logger.debug("line_search: %d backtracks; %.3e -> %.3e; cost %.6e",
                 step_count, first_step, step_length, cost_new)
    return step_length, cost_new, x_new


def _conjugate_gradient(obj, data, theta, tilt, num_iter, step_length):
    """Run conjugate gradient optimization.

    Parameters
    ----------
    obj : (n, n, n) complex64
        Initial volume estimate.
    data : (R, n, n) complex64
        Projection data.
    theta : (R,) float32
        Rotation angles.
    tilt : float
        Tilt angle.
    num_iter : int
        Number of CG iterations.
    step_length : float
        Initial step length for line search.

    Returns
    -------
    obj : (n, n, n) complex64
        Updated volume.
    cost : float
        Final cost value.
    """
    xp = _get_xp(obj)

    def cost_fn(x):
        return cost_function(x, data, theta, tilt)

    cost = None
    grad0 = None
    dir_ = None

    for i in range(num_iter):
        grad1 = gradient(obj, data, theta, tilt)

        dir_ = _direction_dai_yuan(xp, grad1, grad0, dir_)
        grad0 = grad1

        step_length, cost, obj = _line_search(
            cost_fn, obj, dir_, step_length, cost_x=cost,
        )

    if cost is None:
        cost = cost_fn(obj)

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return obj, cost


def reconstruct(data, theta, tilt, volume_shape, n_rounds=5, n_iter_per_round=4):
    """Reconstruct a 3D volume from laminographic projections.

    Uses conjugate gradient descent with Dai-Yuan direction and
    backtracking line search.

    Parameters
    ----------
    data : (R, n, n) complex64
        Projection data.
    theta : (R,) float32
        Rotation angles in radians.
    tilt : float
        Tilt angle in radians. pi/2 for standard tomography.
    volume_shape : tuple of int
        Shape (nz, n, n) of the volume to reconstruct.
    n_rounds : int
        Number of reconstruction rounds.
    n_iter_per_round : int
        Number of CG iterations per round.

    Returns
    -------
    dict
        'obj': reconstructed volume (nz, n, n) complex64
        'costs': list of cost values per round
    """
    data = np.asarray(data, dtype=np.complex64)
    theta = np.asarray(theta, dtype=np.float32)
    tilt_f = np.float32(tilt)

    assert data.ndim == 3, f"data must be 3D, got shape {data.shape}"
    assert theta.ndim == 1, f"theta must be 1D, got shape {theta.shape}"

    # Move to GPU
    data_g = _to_gpu(data)
    theta_g = _to_gpu(theta)
    obj = _to_gpu(np.zeros(volume_shape, dtype=np.complex64))

    # Estimate step length
    step_length = 1.0

    all_costs = []

    for r in range(n_rounds):
        if r == 0:
            # Use a small initial object for step estimation
            test_obj = obj + _get_xp(obj).ones_like(obj) * 1e-6
            step_length = _estimate_step_length(test_obj, theta_g, tilt_f)

        obj, cost = _conjugate_gradient(
            obj, data_g, theta_g, tilt_f,
            num_iter=n_iter_per_round,
            step_length=step_length,
        )

        all_costs.append(float(cost))
        logger.info(f"Round {r + 1}/{n_rounds}: cost = {cost:.6f}")

    return {
        'obj': _to_cpu(obj),
        'costs': all_costs,
    }
