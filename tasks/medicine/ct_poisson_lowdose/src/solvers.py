"""
CT reconstruction solvers for low-dose (Poisson noise) data.

Provides:
  1. Unweighted TV reconstruction (baseline — ignores Poisson statistics)
  2. PWLS-TV reconstruction with Poisson-derived weights (main method)

Both use the Chambolle-Pock primal-dual algorithm (PDHG), adapted from
ct_sparse_view. The key difference is the weighted data fidelity term:

  Unweighted: (1/2) ||Ax - y||^2  +  lambda * TV(x)
  PWLS:       (1/2) (Ax - y)^T W (Ax - y)  +  lambda * TV(x)

where W = diag(w_i) with w_i = I_i (measured photon counts).

The gradient of the weighted fidelity is: A^T W (Ax - y).

Reference
---------
Chambolle, A. & Pock, T. (2011). A first-order primal-dual algorithm.
Elbakri, I.A. & Fessler, J.A. (2002). Statistical image reconstruction
for polyenergetic X-ray computed tomography. IEEE TMI.
SVMBIR: Bouman et al. (Purdue), https://github.com/cabouman/svmbir
"""

import numpy as np
from src.physics_model import radon_forward, radon_backproject


# ---------------------------------------------------------------------------
# Discrete gradient / divergence (same as ct_sparse_view)
# ---------------------------------------------------------------------------

def gradient_2d(x):
    """Discrete gradient of a 2D image.

    Returns
    -------
    grad : ndarray, (2, H, W)
        grad[0] = forward difference along rows
        grad[1] = forward difference along cols
    """
    grad = np.zeros((2,) + x.shape, dtype=x.dtype)
    grad[0, :-1, :] = x[1:, :] - x[:-1, :]
    grad[1, :, :-1] = x[:, 1:] - x[:, :-1]
    return grad


def divergence_2d(p):
    """Negative divergence (adjoint of gradient).

    Parameters
    ----------
    p : ndarray, (2, H, W)

    Returns
    -------
    div : ndarray, (H, W)
    """
    H, W = p.shape[1], p.shape[2]
    div = np.zeros((H, W), dtype=p.dtype)

    div[0, :] = p[0, 0, :]
    div[1:-1, :] += p[0, 1:-1, :] - p[0, :-2, :]
    div[-1, :] -= p[0, -2, :]

    div[:, 0] += p[1, :, 0]
    div[:, 1:-1] += p[1, :, 1:-1] - p[1, :, :-2]
    div[:, -1] -= p[1, :, -2]

    return div


# ---------------------------------------------------------------------------
# Weighted PDHG solver (Chambolle-Pock with weighted data fidelity)
# ---------------------------------------------------------------------------

def estimate_operator_norm(angles, num_channels, num_rows, num_cols,
                           max_iter=10):
    """Estimate ||A||^2 via power iteration for step size selection.

    Parameters
    ----------
    angles, num_channels, num_rows, num_cols : as for radon_forward/backproject
    max_iter : int

    Returns
    -------
    op_norm_sq : float
        Estimated ||A^T A|| (largest eigenvalue of A^T A).
    """
    x = np.random.randn(num_rows, num_cols)
    x /= np.linalg.norm(x)
    for _ in range(max_iter):
        Ax = radon_forward(x, angles, num_channels)
        ATAx = radon_backproject(Ax, angles, num_rows, num_cols)
        op_norm_sq = np.linalg.norm(ATAx)
        if op_norm_sq > 0:
            x = ATAx / op_norm_sq
    return float(op_norm_sq)


def pwls_tv_reconstruction(sinogram, angles, weights, num_rows, num_cols,
                            lam=0.005, n_iter=200, positivity=True):
    """Penalized Weighted Least Squares reconstruction with TV prior.

    Solves: min_x  (1/2)(Ax - y)^T W (Ax - y)  +  lam * TV(x)

    using proximal gradient descent (ISTA) with step size from power
    iteration on A^T A. The TV proximal is computed via Chambolle's
    dual projection (inner iterations).

    Parameters
    ----------
    sinogram : ndarray, (num_views, num_channels)
        Post-log sinogram measurements.
    angles : ndarray, (num_views,)
        Projection angles in radians.
    weights : ndarray, (num_views, num_channels) or None
        Per-measurement weights. For Poisson noise: w_i = I_i (photon counts).
        If None, uses uniform weights (unweighted reconstruction).
    num_rows, num_cols : int
        Output image dimensions.
    lam : float
        TV regularization strength.
    n_iter : int
        Number of iterations.
    positivity : bool
        Enforce non-negativity.

    Returns
    -------
    x : ndarray, (num_rows, num_cols)
        Reconstructed image.
    """
    num_channels = sinogram.shape[1]

    # Default: uniform weights
    if weights is None:
        weights = np.ones_like(sinogram)

    # Normalise weights
    w_max = np.max(weights)
    W = weights / w_max if w_max > 0 else weights

    # Estimate step size via power iteration on A^T A
    op_norm_sq = estimate_operator_norm(angles, num_channels, num_rows, num_cols)
    step_size = 1.0 / op_norm_sq

    # Initialize with zeros
    x = np.zeros((num_rows, num_cols), dtype=np.float64)

    for it in range(n_iter):
        # Gradient of weighted data fidelity: A^T W (Ax - y)
        Ax = radon_forward(x, angles, num_channels)
        residual = Ax - sinogram
        weighted_residual = W * residual
        grad = radon_backproject(weighted_residual, angles, num_rows, num_cols)

        # Gradient step
        x = x - step_size * grad

        # TV proximal via Chambolle's dual projection (inner iterations)
        z = x.copy()
        p = np.zeros((2, num_rows, num_cols), dtype=np.float64)
        tv_tau = 0.25
        for _ in range(20):
            grad_z = gradient_2d(z)
            p = p + tv_tau * grad_z
            p = p / np.maximum(1.0, np.sqrt(p[0]**2 + p[1]**2)[np.newaxis, :, :])
            z = x - lam * step_size * divergence_2d(p)
            if positivity:
                z = np.maximum(z, 0)
        x = z

        if positivity:
            x = np.maximum(x, 0)

    return x


def unweighted_tv_reconstruction(sinogram, angles, num_rows, num_cols,
                                  lam=0.005, n_iter=200, positivity=True):
    """Unweighted TV reconstruction (baseline, ignores Poisson statistics).

    Same as pwls_tv_reconstruction but with uniform weights.

    Parameters
    ----------
    sinogram : ndarray, (num_views, num_channels)
    angles : ndarray, (num_views,)
    num_rows, num_cols : int
    lam : float
    n_iter : int
    positivity : bool

    Returns
    -------
    x : ndarray, (num_rows, num_cols)
    """
    return pwls_tv_reconstruction(
        sinogram, angles, weights=None,
        num_rows=num_rows, num_cols=num_cols,
        lam=lam, n_iter=n_iter, positivity=positivity,
    )
