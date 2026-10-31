"""
MLEM and OSEM solvers for PET reconstruction.

MLEM (Maximum Likelihood Expectation Maximization) is the standard
iterative algorithm for Poisson emission tomography. It maximizes
the Poisson log-likelihood via multiplicative updates.

MLEM update:
    x^{k+1} = x^k / (A^T 1) * A^T (y / (A x^k + r))

where:
    x = activity image (unknown)
    y = measured sinogram (Poisson counts)
    A = system matrix (forward projection)
    A^T = back-projection (adjoint)
    r = background (randoms + scatter)
    A^T 1 = sensitivity image

OSEM accelerates MLEM by using ordered subsets of projection angles.
"""

import numpy as np
from src.physics_model import (
    pet_forward_project,
    pet_back_project,
    compute_sensitivity_image,
)


def solve_mlem(sinogram, theta, N, n_iter=50, background=None,
               x_init=None, verbose=True):
    """MLEM reconstruction for PET.

    Parameters
    ----------
    sinogram : np.ndarray
        Measured sinogram (Poisson counts), shape (n_radial, n_angles).
    theta : np.ndarray
        Projection angles in degrees, shape (n_angles,).
    N : int
        Output image size (N x N).
    n_iter : int
        Number of MLEM iterations.
    background : np.ndarray or None
        Background sinogram (randoms + scatter), same shape as sinogram.
        If None, no background correction.
    x_init : np.ndarray or None
        Initial image estimate. If None, uses uniform positive image.
    verbose : bool
        Print progress.

    Returns
    -------
    x : np.ndarray
        Reconstructed activity image, shape (N, N).
    log_likelihood_history : list of float
        Poisson log-likelihood at each iteration.
    """
    # Sensitivity image: A^T 1
    sensitivity = compute_sensitivity_image(theta, N)
    sensitivity = np.maximum(sensitivity, 1e-10)

    # Initialize
    if x_init is not None:
        x = x_init.copy()
    else:
        x = np.ones((N, N), dtype=np.float64)
    x = np.maximum(x, 1e-10)

    if background is None:
        background = np.zeros_like(sinogram)

    log_likelihood_history = []

    for it in range(n_iter):
        # E-step: compute expected sinogram
        expected = pet_forward_project(x, theta) + background
        expected = np.maximum(expected, 1e-10)

        # Ratio: y / (Ax + r)
        ratio = sinogram / expected

        # M-step: back-project ratio and multiply
        correction = pet_back_project(ratio, theta, N)
        x = x / sensitivity * correction

        # Enforce non-negativity
        x = np.maximum(x, 1e-10)

        # Compute Poisson log-likelihood: sum(y*log(Ax+r) - (Ax+r))
        expected_clipped = np.maximum(expected, 1e-10)
        ll = np.sum(sinogram * np.log(expected_clipped) - expected_clipped)
        log_likelihood_history.append(ll)

        if verbose and (it + 1) % 10 == 0:
            print(f"    MLEM iter {it+1}/{n_iter}: log-likelihood={ll:.2f}")

    return x, log_likelihood_history


def solve_osem(sinogram, theta, N, n_iter=10, n_subsets=6,
               background=None, x_init=None, verbose=True):
    """OSEM (Ordered Subsets EM) reconstruction for PET.

    Accelerates MLEM by dividing projection angles into subsets
    and updating the image after each subset.

    Parameters
    ----------
    sinogram : np.ndarray
        Measured sinogram, shape (n_radial, n_angles).
    theta : np.ndarray
        Projection angles in degrees, shape (n_angles,).
    N : int
        Output image size.
    n_iter : int
        Number of full iterations (each uses all subsets).
    n_subsets : int
        Number of ordered subsets.
    background : np.ndarray or None
        Background sinogram.
    x_init : np.ndarray or None
        Initial image.
    verbose : bool

    Returns
    -------
    x : np.ndarray
        Reconstructed image, shape (N, N).
    log_likelihood_history : list of float
        Log-likelihood after each full iteration.
    """
    n_angles = len(theta)

    # Create subset indices (interleaved for balanced angular coverage)
    subset_indices = [list(range(s, n_angles, n_subsets)) for s in range(n_subsets)]

    # Initialize
    if x_init is not None:
        x = x_init.copy()
    else:
        x = np.ones((N, N), dtype=np.float64)
    x = np.maximum(x, 1e-10)

    if background is None:
        background = np.zeros_like(sinogram)

    log_likelihood_history = []

    for it in range(n_iter):
        for s in range(n_subsets):
            idx = subset_indices[s]
            theta_sub = theta[idx]
            sino_sub = sinogram[:, idx]
            bg_sub = background[:, idx]

            # Subset sensitivity
            sensitivity_sub = compute_sensitivity_image(theta_sub, N)
            sensitivity_sub = np.maximum(sensitivity_sub, 1e-10)

            # E-step
            expected_sub = pet_forward_project(x, theta_sub) + bg_sub
            expected_sub = np.maximum(expected_sub, 1e-10)

            # Ratio
            ratio_sub = sino_sub / expected_sub

            # M-step
            correction_sub = pet_back_project(ratio_sub, theta_sub, N)
            x = x / sensitivity_sub * correction_sub
            x = np.maximum(x, 1e-10)

        # Log-likelihood after full iteration
        expected_full = pet_forward_project(x, theta) + background
        expected_full = np.maximum(expected_full, 1e-10)
        ll = np.sum(sinogram * np.log(expected_full) - expected_full)
        log_likelihood_history.append(ll)

        if verbose:
            print(f"    OSEM iter {it+1}/{n_iter}: log-likelihood={ll:.2f}")

    return x, log_likelihood_history
