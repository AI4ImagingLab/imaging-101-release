"""
Solvers for sparse-view CT reconstruction.

Implements TV-regularized iterative reconstruction using the Chambolle-Pock
primal-dual algorithm (a.k.a. PDHG). This is well-suited for CT because:
  - The Radon transform is a linear operator
  - TV regularization promotes piecewise-constant structure
  - The algorithm handles non-smooth TV without smoothing approximations

The Chambolle-Pock algorithm solves:
    min_x  (1/2)||Ax - b||^2  +  lambda * TV(x)
where A is the Radon transform, b is the measured sinogram, and TV(x) is
the isotropic total variation.

Reference
---------
Chambolle, A. & Pock, T. (2011). A first-order primal-dual algorithm for
convex problems with applications to imaging. JMIV.
"""

import numpy as np
from src.physics_model import radon_transform, filtered_back_projection


def gradient_2d(x):
    """Compute discrete gradient of a 2D image.

    Parameters
    ----------
    x : np.ndarray, shape (H, W)

    Returns
    -------
    grad : np.ndarray, shape (2, H, W)
        grad[0] = forward difference along axis 0 (rows)
        grad[1] = forward difference along axis 1 (cols)
    """
    grad = np.zeros((2,) + x.shape, dtype=x.dtype)
    grad[0, :-1, :] = x[1:, :] - x[:-1, :]
    grad[1, :, :-1] = x[:, 1:] - x[:, :-1]
    return grad


def divergence_2d(p):
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

    # adjoint of forward difference along axis 0
    div[0, :] = p[0, 0, :]
    div[1:-1, :] += p[0, 1:-1, :] - p[0, :-2, :]
    div[-1, :] -= p[0, -2, :]

    # adjoint of forward difference along axis 1
    div[:, 0] += p[1, :, 0]
    div[:, 1:-1] += p[1, :, 1:-1] - p[1, :, :-2]
    div[:, -1] -= p[1, :, -2]

    return div


def tv_reconstruction(sinogram, angles_deg, output_size, lam=0.01,
                       n_iter=300, positivity=True):
    """TV-regularized CT reconstruction using Chambolle-Pock (PDHG).

    Solves: min_x (1/2)||Ax - b||^2 + lam * TV(x)

    Parameters
    ----------
    sinogram : np.ndarray, shape (n_det, n_angles)
        Measured sinogram.
    angles_deg : np.ndarray, shape (n_angles,)
        Projection angles in degrees.
    output_size : int
        Size of the reconstructed image (output_size x output_size).
    lam : float
        TV regularization weight.
    n_iter : int
        Number of iterations.
    positivity : bool
        If True, enforce non-negativity constraint.

    Returns
    -------
    x : np.ndarray, shape (output_size, output_size)
        Reconstructed image.
    loss_history : list of float
        Data fidelity loss at each iteration.
    """
    # Initialize with FBP
    x = filtered_back_projection(sinogram, angles_deg, output_size=output_size,
                                  filter_name="ramp")
    if positivity:
        x = np.maximum(x, 0)

    # Dual variable for TV
    p = np.zeros((2, output_size, output_size), dtype=np.float64)

    # Step sizes (conservative choice for convergence)
    tau = 0.01     # primal step size
    sigma = 0.5    # dual step size
    theta = 1.0    # extrapolation parameter

    x_bar = x.copy()
    loss_history = []

    for k in range(n_iter):
        # Dual update: p <- prox_{sigma * lam * TV*}(p + sigma * grad(x_bar))
        grad_xbar = gradient_2d(x_bar)
        p = p + sigma * grad_xbar
        # Projection (proximal of conjugate of lam*TV = indicator of ||.||<=lam)
        p = p / np.maximum(1.0, np.sqrt(p[0]**2 + p[1]**2)[np.newaxis, :, :] / lam)

        # Primal update: x <- prox_{tau * f}(x + tau * div(p))
        x_old = x.copy()
        div_p = divergence_2d(p)

        # Gradient of data fidelity: A^T(Ax - b)
        Ax = radon_transform(x, angles_deg)
        residual = Ax - sinogram
        backproj = filtered_back_projection(
            residual, angles_deg, output_size=output_size, filter_name=None,
        ) * (np.pi / (2 * len(angles_deg)))

        x = x - tau * backproj + tau * div_p

        if positivity:
            x = np.maximum(x, 0)

        # Extrapolation
        x_bar = x + theta * (x - x_old)

        # Track data fidelity loss
        data_loss = 0.5 * np.sum(residual**2)
        loss_history.append(float(data_loss))

    return x, loss_history
