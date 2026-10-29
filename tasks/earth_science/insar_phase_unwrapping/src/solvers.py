"""Inverse solver: ADMM-based sparse phase unwrapping.

Implements the algorithm from:
    Chartrand et al., "Exploiting Sparsity for Phase Unwrapping", IGARSS 2019.

The algorithm solves:
    min_Phi ||D*Phi - phi||_0
by ADMM splitting with a nonconvex sparsity-promoting penalty G0.
"""

import numpy as np
from .physics_model import (
    apply_gradient_x,
    apply_gradient_y,
    apply_divergence,
    make_laplace_kernel,
    solve_poisson_dct,
)
from .preprocessing import est_wrapped_gradient


def p_shrink(X, lmbda=1, p=0, epsilon=0):
    """p-shrinkage operator (Eq. 11 in the paper).

    For p=0, this implements the G0 penalty shrinkage:
        w = max(|x| - 1/|x|, 0) * sign(x)

    Parameters
    ----------
    X : ndarray, shape (2, rows, columns)
        Stacked x and y components.
    lmbda : float
        Splitting parameter.
    p : float
        Shrinkage exponent (0 for G0 penalty).
    epsilon : float
        Regularization for numerical stability.

    Returns
    -------
    ndarray, shape (2, rows, columns)
    """
    mag = np.sqrt(np.sum(X ** 2, axis=0))
    nonzero = np.where(mag == 0.0, 1.0, mag)
    mag = (
        np.maximum(
            mag - lmbda ** (2.0 - p)
            * (nonzero ** 2 + epsilon) ** (p / 2.0 - 0.5),
            0,
        )
        / nonzero
    )
    return mag * X


def make_congruent(unwrapped, wrapped):
    """Adjust unwrapped phase to be congruent with wrapped phase.

    Rounds to the nearest integer ambiguity so that:
        unwrapped_out = wrapped + 2*pi*k  for integer k at each pixel.

    Parameters
    ----------
    unwrapped : ndarray
    wrapped : ndarray

    Returns
    -------
    ndarray
    """
    k = np.round((unwrapped - wrapped) / (2 * np.pi))
    return wrapped + 2 * np.pi * k


def unwrap_phase(
    f_wrapped,
    phi_x=None,
    phi_y=None,
    max_iters=500,
    tol=np.pi / 5,
    lmbda=1,
    p=0,
    c=1.3,
    dtype="float32",
    debug=False,
    congruent=False,
):
    """Unwrap interferogram phase using ADMM sparse optimization.

    Parameters
    ----------
    f_wrapped : ndarray, shape (rows, columns)
        Wrapped phase image.
    phi_x, phi_y : ndarray, optional
        Pre-estimated wrapped gradients. Computed if not provided.
    max_iters : int
        Maximum ADMM iterations.
    tol : float
        Convergence tolerance (max pixel change in radians).
    lmbda : float
        ADMM splitting parameter.
    p : float
        Shrinkage exponent (0 = G0 penalty from paper).
    c : float
        Lagrange multiplier acceleration constant.
    dtype : str
        Array dtype.
    debug : bool
        Print iteration diagnostics.
    congruent : bool
        If True, snap result to nearest 2*pi*k offset from input.

    Returns
    -------
    F : ndarray, shape (rows, columns)
        Unwrapped phase.
    n_iters : int
        Number of iterations performed.
    """
    rows, columns = f_wrapped.shape
    if dtype is not None:
        f_wrapped = f_wrapped.astype(dtype)

    if phi_x is None or phi_y is None:
        phi_x, phi_y = est_wrapped_gradient(f_wrapped, dtype=dtype)

    # Initialize variables
    Lambda_x = np.zeros_like(phi_x, dtype=dtype)
    Lambda_y = np.zeros_like(phi_y, dtype=dtype)
    w_x = np.zeros_like(phi_x, dtype=dtype)
    w_y = np.zeros_like(phi_y, dtype=dtype)
    F_old = np.zeros_like(f_wrapped)
    K = make_laplace_kernel(rows, columns, dtype=dtype)

    n_iters = 0
    for iteration in range(max_iters):
        # Step 1: Solve linear system via DCT (Eq. 10)
        RHS = apply_divergence(
            w_x + phi_x - Lambda_x,
            w_y + phi_y - Lambda_y)
        F = solve_poisson_dct(RHS, K)

        # Step 2: Compute gradients of current estimate
        Fx = apply_gradient_x(F)
        Fy = apply_gradient_y(F)

        # Step 3: Shrinkage step (Eq. 11)
        stacked = np.stack(
            (Fx - phi_x + Lambda_x, Fy - phi_y + Lambda_y),
            axis=0)
        shrunk = p_shrink(stacked, lmbda=lmbda, p=p, epsilon=0)
        w_x, w_y = shrunk[0], shrunk[1]

        # Step 4: Update Lagrange multipliers
        Lambda_x = Lambda_x + c * (Fx - phi_x - w_x)
        Lambda_y = Lambda_y + c * (Fy - phi_y - w_y)

        change = float(np.max(np.abs(F - F_old)))
        n_iters = iteration + 1
        if debug:
            print(f"Iteration:{iteration} change={change}")
        if change < tol or np.isnan(change):
            break
        F_old = F

    if debug:
        print(f"Finished after {n_iters} iterations with change={change}")

    if congruent:
        F = make_congruent(F, f_wrapped)
    return F, n_iters
