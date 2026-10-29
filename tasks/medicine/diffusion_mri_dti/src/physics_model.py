"""
Diffusion MRI signal model (Stejskal-Tanner equation).

Forward model:
    S(g, b) = S0 * exp(-b * g^T D g)

where:
    D is a 3x3 symmetric positive-definite diffusion tensor
    g is the gradient direction unit vector
    b is the b-value (diffusion weighting, s/mm^2)
    S0 is the non-diffusion-weighted signal
"""

import numpy as np


def tensor_from_elements(Dxx, Dxy, Dxz, Dyy, Dyz, Dzz):
    """Construct 3x3 symmetric tensor(s) from 6 independent elements.

    Parameters
    ----------
    Dxx, Dxy, Dxz, Dyy, Dyz, Dzz : np.ndarray
        Tensor elements, all same shape (...,).

    Returns
    -------
    D : np.ndarray
        Symmetric diffusion tensor(s), shape (..., 3, 3).
    """
    shape = np.broadcast_shapes(*(np.shape(x) for x in [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]))
    D = np.zeros(shape + (3, 3), dtype=np.float64)
    D[..., 0, 0] = Dxx
    D[..., 0, 1] = Dxy
    D[..., 0, 2] = Dxz
    D[..., 1, 0] = Dxy
    D[..., 1, 1] = Dyy
    D[..., 1, 2] = Dyz
    D[..., 2, 0] = Dxz
    D[..., 2, 1] = Dyz
    D[..., 2, 2] = Dzz
    return D


def elements_from_tensor(D):
    """Extract 6 independent elements from symmetric 3x3 tensor(s).

    Parameters
    ----------
    D : np.ndarray
        Symmetric tensor(s), shape (..., 3, 3).

    Returns
    -------
    elements : np.ndarray
        Tensor elements [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], shape (..., 6).
    """
    return np.stack([
        D[..., 0, 0], D[..., 0, 1], D[..., 0, 2],
        D[..., 1, 1], D[..., 1, 2], D[..., 2, 2],
    ], axis=-1)


def tensor_from_eig(eigenvalues, eigenvectors):
    """Construct diffusion tensor from eigendecomposition.

    D = V @ diag(lambda) @ V^T

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues (lambda1, lambda2, lambda3), shape (..., 3).
        Convention: lambda1 >= lambda2 >= lambda3.
    eigenvectors : np.ndarray
        Eigenvectors as columns, shape (..., 3, 3).
        eigenvectors[..., :, i] is the i-th eigenvector.

    Returns
    -------
    D : np.ndarray
        Symmetric diffusion tensor, shape (..., 3, 3).
    """
    # D = V @ diag(evals) @ V^T
    evals_diag = np.zeros(eigenvalues.shape[:-1] + (3, 3), dtype=np.float64)
    evals_diag[..., 0, 0] = eigenvalues[..., 0]
    evals_diag[..., 1, 1] = eigenvalues[..., 1]
    evals_diag[..., 2, 2] = eigenvalues[..., 2]
    D = eigenvectors @ evals_diag @ np.swapaxes(eigenvectors, -2, -1)
    return D


def build_design_matrix(bvals, bvecs):
    """Build the design matrix for the linearized Stejskal-Tanner equation.

    The linearized model is:
        ln(S_i) = ln(S0) - b_i * g_i^T D g_i

    Rewritten as:
        ln(S_i) = [1, -b*gx^2, -2b*gx*gy, -2b*gx*gz, -b*gy^2, -2b*gy*gz, -b*gz^2] @ [ln(S0), Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]

    Parameters
    ----------
    bvals : np.ndarray
        b-values in s/mm^2, shape (N_volumes,).
    bvecs : np.ndarray
        Gradient directions (unit vectors), shape (N_volumes, 3).
        For b=0 volumes, bvecs should be [0, 0, 0].

    Returns
    -------
    B : np.ndarray
        Design matrix, shape (N_volumes, 7).
        Columns: [1, -b*gx^2, -2b*gx*gy, -2b*gx*gz, -b*gy^2, -2b*gy*gz, -b*gz^2]
    """
    bvals = np.asarray(bvals, dtype=np.float64)
    bvecs = np.asarray(bvecs, dtype=np.float64)
    N = len(bvals)

    gx, gy, gz = bvecs[:, 0], bvecs[:, 1], bvecs[:, 2]

    B = np.zeros((N, 7), dtype=np.float64)
    B[:, 0] = 1.0                       # ln(S0) coefficient
    B[:, 1] = -bvals * gx * gx          # Dxx
    B[:, 2] = -2.0 * bvals * gx * gy    # Dxy
    B[:, 3] = -2.0 * bvals * gx * gz    # Dxz
    B[:, 4] = -bvals * gy * gy          # Dyy
    B[:, 5] = -2.0 * bvals * gy * gz    # Dyz
    B[:, 6] = -bvals * gz * gz          # Dzz
    return B


def stejskal_tanner_signal(S0, D, bvals, bvecs):
    """Compute diffusion-weighted signal using Stejskal-Tanner equation.

    S_i = S0 * exp(-b_i * g_i^T D g_i)

    Parameters
    ----------
    S0 : np.ndarray
        Non-diffusion-weighted signal, shape (...,).
    D : np.ndarray
        Diffusion tensor(s), shape (..., 3, 3).
    bvals : np.ndarray
        b-values in s/mm^2, shape (N_volumes,).
    bvecs : np.ndarray
        Gradient directions, shape (N_volumes, 3).

    Returns
    -------
    signal : np.ndarray
        DWI signal, shape (..., N_volumes).
    """
    bvals = np.asarray(bvals, dtype=np.float64)
    bvecs = np.asarray(bvecs, dtype=np.float64)
    S0 = np.asarray(S0, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    N_volumes = len(bvals)
    signal = np.zeros(S0.shape + (N_volumes,), dtype=np.float64)

    for i in range(N_volumes):
        g = bvecs[i]  # (3,)
        # g^T D g = quadratic form
        adc = g @ D @ g  # apparent diffusion coefficient along direction g
        # adc has shape (...,) via broadcasting: D is (...,3,3), g is (3,)
        # For batch: need einsum
        if D.ndim > 2:
            adc = np.einsum('i,...ij,j->...', g, D, g)
        signal[..., i] = S0 * np.exp(-bvals[i] * adc)

    return signal


def add_rician_noise(signal, sigma, rng=None):
    """Add Rician noise to magnitude MRI signal.

    Parameters
    ----------
    signal : np.ndarray
        Clean signal, any shape.
    sigma : float
        Noise standard deviation per channel.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    noisy : np.ndarray
        Noisy magnitude signal, same shape as input.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise_real = rng.normal(0, sigma, signal.shape)
    noise_imag = rng.normal(0, sigma, signal.shape)
    noisy = np.sqrt((signal + noise_real) ** 2 + noise_imag ** 2)
    return noisy


def compute_fa(eigenvalues):
    """Compute fractional anisotropy from tensor eigenvalues.

    FA = sqrt(3/2) * sqrt(sum((lambda_i - MD)^2) / sum(lambda_i^2))

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues, shape (..., 3).

    Returns
    -------
    fa : np.ndarray
        Fractional anisotropy in [0, 1], shape (...,).
    """
    md = np.mean(eigenvalues, axis=-1, keepdims=True)
    numerator = np.sum((eigenvalues - md) ** 2, axis=-1)
    denominator = np.sum(eigenvalues ** 2, axis=-1)

    with np.errstate(divide='ignore', invalid='ignore'):
        fa = np.sqrt(1.5 * numerator / denominator)

    fa = np.where(np.isfinite(fa), fa, 0.0)
    fa = np.clip(fa, 0.0, 1.0)
    return fa


def compute_md(eigenvalues):
    """Compute mean diffusivity from tensor eigenvalues.

    MD = (lambda1 + lambda2 + lambda3) / 3

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues, shape (..., 3).

    Returns
    -------
    md : np.ndarray
        Mean diffusivity, shape (...,). Same units as eigenvalues.
    """
    return np.mean(eigenvalues, axis=-1)
