"""
Iterative reconstruction solvers for fan-beam CT.

TV-regularized reconstruction using Chambolle-Pock (PDHG) algorithm,
adapted from the ct_sparse_view task to use fan-beam operators.
"""

import numpy as np
from src.physics_model import (
    fan_beam_forward_vectorized,
    fan_beam_backproject,
    fan_beam_fbp,
)


def _gradient_2d(x):
    """Discrete gradient operator (forward differences).

    Parameters
    ----------
    x : np.ndarray, shape (N, N)

    Returns
    -------
    grad : np.ndarray, shape (2, N, N)
        grad[0] = horizontal diff, grad[1] = vertical diff.
    """
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:, :-1] = x[:, 1:] - x[:, :-1]
    gy[:-1, :] = x[1:, :] - x[:-1, :]
    return np.stack([gx, gy], axis=0)


def _divergence_2d(p):
    """Discrete divergence (negative adjoint of gradient).

    Parameters
    ----------
    p : np.ndarray, shape (2, N, N)

    Returns
    -------
    div : np.ndarray, shape (N, N)
    """
    px, py = p[0], p[1]
    dx = np.zeros_like(px)
    dy = np.zeros_like(py)
    dx[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    dx[:, 0] = px[:, 0]
    dx[:, -1] = -px[:, -2]
    dy[1:-1, :] = py[1:-1, :] - py[:-2, :]
    dy[0, :] = py[0, :]
    dy[-1, :] = -py[-2, :]
    return dx + dy


def _prox_l1_norm(p, sigma):
    """Proximal operator for sigma * ||p||_1 (pointwise L2 norm on gradient).

    Projects onto L2 ball of radius sigma at each pixel.
    """
    norm = np.sqrt(p[0] ** 2 + p[1] ** 2)
    norm = np.maximum(norm, 1e-10)
    factor = np.maximum(norm, sigma) / norm
    return p / factor[np.newaxis, ...]


def solve_tv_pdhg(sinogram, geo, lam=0.01, n_iter=200, positivity=True,
                  x_init=None, verbose=True):
    """TV-regularized fan-beam CT reconstruction via Chambolle-Pock (PDHG).

    Solves: min_x  (1/2) ||A x - b||^2  +  lam * TV(x)

    where A is the fan-beam forward operator and TV is the isotropic
    total variation.

    Parameters
    ----------
    sinogram : np.ndarray
        Measured sinogram, shape (n_angles, n_det).
    geo : dict
        Fan-beam geometry.
    lam : float
        TV regularization weight.
    n_iter : int
        Number of PDHG iterations.
    positivity : bool
        Enforce non-negativity constraint.
    x_init : np.ndarray or None
        Initial reconstruction. If None, uses FBP.
    verbose : bool
        Print progress.

    Returns
    -------
    x : np.ndarray
        Reconstructed image, shape (N, N).
    loss_history : list of float
        Data fidelity loss at each iteration.
    """
    N = geo['N']

    # Initialize with FBP
    if x_init is None:
        short_scan = geo['angle_range'] < 1.9 * np.pi
        x = fan_beam_fbp(sinogram, geo, filter_type='hann', cutoff=0.3,
                         short_scan=short_scan)
        if positivity:
            x = np.maximum(x, 0)
    else:
        x = x_init.copy()

    # PDHG parameters
    # Step sizes: tau * sigma * ||K||^2 < 1
    # ||K|| includes both A (ray transform) and gradient
    tau = 0.005
    sigma_dual = 0.005
    theta = 1.0

    # Dual variables
    p = np.zeros((2, N, N), dtype=np.float64)  # for gradient/TV
    q = np.zeros_like(sinogram)                 # for data fidelity

    x_bar = x.copy()
    loss_history = []

    for it in range(n_iter):
        # --- Dual update for data fidelity ---
        Ax_bar = fan_beam_forward_vectorized(x_bar, geo)
        q = q + sigma_dual * (Ax_bar - sinogram)
        # Proximal: for L2 data fidelity, prox = q / (1 + sigma_dual)
        q = q / (1 + sigma_dual)

        # --- Dual update for TV ---
        grad_x_bar = _gradient_2d(x_bar)
        p = p + sigma_dual * grad_x_bar
        p = _prox_l1_norm(p, lam)

        # --- Primal update ---
        x_old = x.copy()
        At_q = fan_beam_backproject(q, geo)
        div_p = _divergence_2d(p)
        x = x - tau * (At_q - div_p)

        if positivity:
            x = np.maximum(x, 0)

        # Over-relaxation
        x_bar = x + theta * (x - x_old)

        # Loss
        residual = fan_beam_forward_vectorized(x, geo) - sinogram
        loss = 0.5 * np.sum(residual ** 2)
        loss_history.append(loss)

        if verbose and (it + 1) % 50 == 0:
            print(f"    PDHG iter {it+1}/{n_iter}: loss={loss:.6f}")

    return x, loss_history
