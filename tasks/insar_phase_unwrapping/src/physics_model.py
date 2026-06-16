"""Forward model: discrete gradient operators and DCT-based linear solver.

The forward model for phase unwrapping is the discrete gradient operator D.
Given an unwrapped phase field Phi, the forward model produces gradients:
    D * Phi = [Dx * Phi; Dy * Phi]

The inverse problem is: given wrapped gradients phi, find Phi such that
D * Phi is as close to phi as possible (in a sparsity-promoting sense).

The linear system (D^T D) Phi = D^T b is solved efficiently via the
Discrete Cosine Transform (DCT), exploiting the fact that D^T D (the
Laplacian) is diagonalized by the DCT under Neumann boundary conditions.
"""

import numpy as np
from scipy.fft import dctn, idctn


def apply_gradient_x(arr):
    """Forward difference along columns (axis=1) with Neumann BCs.

    Parameters
    ----------
    arr : ndarray, shape (rows, columns)

    Returns
    -------
    ndarray, shape (rows, columns)
        Last column is zero (Neumann BC).
    """
    return np.concatenate([
        arr[:, 1:] - arr[:, :-1],
        np.zeros((arr.shape[0], 1), dtype=arr.dtype)
    ], axis=1)


def apply_gradient_y(arr):
    """Forward difference along rows (axis=0) with Neumann BCs.

    Parameters
    ----------
    arr : ndarray, shape (rows, columns)

    Returns
    -------
    ndarray, shape (rows, columns)
        Last row is zero (Neumann BC).
    """
    return np.concatenate([
        arr[1:, :] - arr[:-1, :],
        np.zeros((1, arr.shape[1]), dtype=arr.dtype)
    ], axis=0)


def apply_divergence(grad_x, grad_y):
    """Divergence (negative adjoint of gradient) for Neumann BCs.

    Parameters
    ----------
    grad_x : ndarray, shape (rows, columns)
    grad_y : ndarray, shape (rows, columns)

    Returns
    -------
    ndarray, shape (rows, columns)
    """
    div_x = np.concatenate([
        -grad_x[:, :1],
        grad_x[:, :-2] - grad_x[:, 1:-1],
        grad_x[:, -2:-1]
    ], axis=1)
    div_y = np.concatenate([
        -grad_y[:1, :],
        grad_y[:-2, :] - grad_y[1:-1, :],
        grad_y[-2:-1, :]
    ], axis=0)
    return div_x + div_y


def make_laplace_kernel(rows, columns, dtype="float32"):
    """Eigenvalues of the Laplacian under Neumann BCs (DCT domain).

    Used to solve (D^T D) Phi = rhs via:
        Phi = IDCT( DCT(rhs) * K )
    where K = 1 / eigenvalues (with K[0,0] = 0).

    Parameters
    ----------
    rows, columns : int
    dtype : str

    Returns
    -------
    K : ndarray, shape (rows, columns)
        Inverse eigenvalues (zero at DC component).
    """
    xi_y = (2 - 2 * np.cos(np.pi * np.arange(rows) / rows)).reshape((-1, 1))
    xi_x = (2 - 2 * np.cos(np.pi * np.arange(columns) / columns)).reshape((1, -1))
    eigvals = xi_y + xi_x
    with np.errstate(divide="ignore"):
        K = np.where(eigvals == 0, 0.0, 1.0 / eigvals)
    return K.astype(dtype)


def solve_poisson_dct(rhs, K):
    """Solve Poisson equation via DCT.

    Solves (D^T D) Phi = rhs using precomputed inverse eigenvalues K.

    Parameters
    ----------
    rhs : ndarray, shape (rows, columns)
    K : ndarray, shape (rows, columns)
        From make_laplace_kernel.

    Returns
    -------
    Phi : ndarray, shape (rows, columns)
    """
    return idctn(dctn(rhs, type=2, norm="ortho", workers=-1) * K,
                 type=2, norm="ortho", workers=-1)
