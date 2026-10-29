"""
Solvers for ultrasound speed-of-sound tomography.

Implements two iterative reconstruction methods for recovering the slowness
field from travel-time sinogram data:

1. SART (Simultaneous Algebraic Reconstruction Technique)
2. TV-regularized Chambolle-Pock (PDHG) reconstruction

Both solve the inverse problem: given t = A @ s + noise, recover s,
where t is the travel-time sinogram, A is the Radon transform (system matrix),
and s is the slowness field.
"""

import numpy as np
from skimage.transform import radon, iradon


# ---------------------------------------------------------------------------
# SART reconstruction
# ---------------------------------------------------------------------------

def sart_reconstruction(sinogram, angles_deg, output_size, n_iter=30,
                        relaxation=0.15):
    """Reconstruct slowness field using SART.

    SART (Simultaneous Algebraic Reconstruction Technique) is an iterative
    row-action method that updates the image by back-projecting the
    normalised residual at each iteration.

    Parameters
    ----------
    sinogram : np.ndarray, shape (n_det, n_angles)
        Measured travel-time sinogram.
    angles_deg : np.ndarray, shape (n_angles,)
        Projection angles in degrees.
    output_size : int
        Size of the reconstructed image.
    n_iter : int
        Number of iterations.
    relaxation : float
        Relaxation parameter (step size).

    Returns
    -------
    x : np.ndarray, shape (output_size, output_size)
        Reconstructed slowness field.
    loss_history : list of float
        Data fidelity loss (0.5 * ||Ax - b||^2) at each iteration.
    """
    # Initialize with FBP
    x = iradon(sinogram, theta=angles_deg, output_size=output_size,
               filter_name="ramp", circle=True)

    scale = np.pi / (2 * len(angles_deg))
    loss_history = []

    for k in range(n_iter):
        # Forward project
        Ax = radon(x, theta=angles_deg, circle=True)
        residual = Ax - sinogram

        # Track loss
        data_loss = 0.5 * np.sum(residual ** 2)
        loss_history.append(float(data_loss))

        # Backproject residual (adjoint)
        grad = iradon(residual, theta=angles_deg, output_size=output_size,
                      filter_name=None, circle=True) * scale

        # Update
        x = x - relaxation * grad

    return x, loss_history


# ---------------------------------------------------------------------------
# TV-PDHG reconstruction (Chambolle-Pock primal-dual)
# ---------------------------------------------------------------------------

def _gradient_2d(x):
    """Compute discrete gradient of a 2D image.

    Parameters
    ----------
    x : np.ndarray, shape (H, W)

    Returns
    -------
    grad : np.ndarray, shape (2, H, W)
        grad[0] = forward difference along rows, grad[1] along cols.
    """
    grad = np.zeros((2,) + x.shape, dtype=x.dtype)
    grad[0, :-1, :] = x[1:, :] - x[:-1, :]
    grad[1, :, :-1] = x[:, 1:] - x[:, :-1]
    return grad


def _divergence_2d(p):
    """Compute negative divergence (adjoint of gradient).

    Parameters
    ----------
    p : np.ndarray, shape (2, H, W)

    Returns
    -------
    div : np.ndarray, shape (H, W)
    """
    H, W = p.shape[1], p.shape[2]
    div = np.zeros((H, W), dtype=p.dtype)

    # Adjoint of forward difference along rows
    div[0, :] = p[0, 0, :]
    div[1:-1, :] += p[0, 1:-1, :] - p[0, :-2, :]
    div[-1, :] -= p[0, -2, :]

    # Adjoint of forward difference along cols
    div[:, 0] += p[1, :, 0]
    div[:, 1:-1] += p[1, :, 1:-1] - p[1, :, :-2]
    div[:, -1] -= p[1, :, -2]

    return div


def _estimate_operator_norm(angles_deg, output_size, max_iter=10):
    """Estimate ||A^T A|| via power iteration for step size selection.

    Adapted from ct_poisson_lowdose/src/solvers.py estimate_operator_norm().

    Parameters
    ----------
    angles_deg : ndarray
    output_size : int
    max_iter : int

    Returns
    -------
    op_norm_sq : float
    """
    x = np.random.randn(output_size, output_size)
    x /= np.linalg.norm(x)
    for _ in range(max_iter):
        Ax = radon(x, theta=angles_deg, circle=True)
        ATAx = iradon(Ax, theta=angles_deg, output_size=output_size,
                       filter_name=None, circle=True)
        op_norm_sq = np.linalg.norm(ATAx)
        if op_norm_sq > 0:
            x = ATAx / op_norm_sq
    return float(op_norm_sq)


def tv_pdhg_reconstruction(sinogram, angles_deg, output_size, lam=1e-7,
                            n_iter=300, positivity=True):
    """TV-regularized reconstruction using Chambolle-Pock (PDHG).

    Solves: min_s  (1/2)||A s - t||^2  +  lam * TV(s)

    where A is the Radon transform, t is the measured sinogram, and
    TV(s) is isotropic total variation.

    Step sizes are derived from the operator norm estimated via power
    iteration, ensuring convergence: tau * sigma * ||A||^2 < 1.

    Parameters
    ----------
    sinogram : np.ndarray, shape (n_det, n_angles)
        Measured travel-time sinogram.
    angles_deg : np.ndarray, shape (n_angles,)
        Projection angles in degrees.
    output_size : int
        Reconstructed image size.
    lam : float
        TV regularization weight.
    n_iter : int
        Number of PDHG iterations.
    positivity : bool
        If True, enforce non-negativity (slowness must be positive).

    Returns
    -------
    x : np.ndarray, shape (output_size, output_size)
        Reconstructed slowness field.
    loss_history : list of float
        Data fidelity loss at each iteration.
    """
    # Estimate operator norm for step size selection
    op_norm = _estimate_operator_norm(angles_deg, output_size)

    # PDHG step sizes satisfying tau * sigma * ||A||^2 < 1
    # Use conservative choice: tau = 1/||A||, sigma = 0.99/||A||
    tau = 1.0 / op_norm
    sigma = 0.99 / op_norm
    theta = 1.0

    # Initialize with FBP
    x = iradon(sinogram, theta=angles_deg, output_size=output_size,
               filter_name="ramp", circle=True)
    if positivity:
        x = np.maximum(x, 0)

    # Dual variable for TV
    p = np.zeros((2, output_size, output_size), dtype=np.float64)

    x_bar = x.copy()
    loss_history = []

    for k in range(n_iter):
        # Dual update: p <- proj_{||.||_inf <= lam}(p + sigma * grad(x_bar))
        grad_xbar = _gradient_2d(x_bar)
        p = p + sigma * grad_xbar
        p = p / np.maximum(1.0, np.sqrt(p[0]**2 + p[1]**2)[np.newaxis, :, :] / lam)

        # Primal update
        x_old = x.copy()
        div_p = _divergence_2d(p)

        # Gradient of data fidelity: A^T(Ax - b)
        Ax = radon(x, theta=angles_deg, circle=True)
        residual = Ax - sinogram
        backproj = iradon(residual, theta=angles_deg, output_size=output_size,
                          filter_name=None, circle=True)

        x = x - tau * backproj + tau * div_p

        if positivity:
            x = np.maximum(x, 0)

        # Extrapolation
        x_bar = x + theta * (x - x_old)

        # Track data fidelity loss
        data_loss = 0.5 * np.sum(residual**2)
        loss_history.append(float(data_loss))

    return x, loss_history
