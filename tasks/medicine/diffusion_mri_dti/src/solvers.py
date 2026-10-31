"""
Diffusion tensor fitting solvers.

Two methods:
1. Ordinary Least Squares (OLS): fast baseline via linearized Stejskal-Tanner
2. Weighted Least Squares (WLS): improved accuracy by weighting with predicted signal
"""

import numpy as np
from src.physics_model import (
    build_design_matrix,
    tensor_from_elements,
    compute_fa,
    compute_md,
)


def fit_dti_ols(dwi_signal, bvals, bvecs, mask=None):
    """Estimate diffusion tensors via ordinary least squares.

    Linearizes the Stejskal-Tanner equation:
        ln(S_i) = ln(S0) - b_i * g_i^T D g_i

    and solves via standard linear least squares for all masked voxels.

    Parameters
    ----------
    dwi_signal : np.ndarray
        DWI signal, shape (Ny, Nx, N_volumes).
    bvals : np.ndarray
        b-values in s/mm^2, shape (N_volumes,).
    bvecs : np.ndarray
        Gradient directions, shape (N_volumes, 3).
    mask : np.ndarray or None
        Boolean tissue mask, shape (Ny, Nx). If None, all pixels are fit.

    Returns
    -------
    tensor_elems : np.ndarray
        Fitted tensor elements [Dxx,Dxy,Dxz,Dyy,Dyz,Dzz], shape (Ny, Nx, 6).
    S0_map : np.ndarray
        Estimated S0, shape (Ny, Nx).
    """
    bvals = np.asarray(bvals, dtype=np.float64)
    bvecs = np.asarray(bvecs, dtype=np.float64)
    dwi_signal = np.asarray(dwi_signal, dtype=np.float64)
    Ny, Nx, N_volumes = dwi_signal.shape

    tensor_elems = np.zeros((Ny, Nx, 6), dtype=np.float64)
    S0_map = np.zeros((Ny, Nx), dtype=np.float64)

    if mask is None:
        mask = np.ones((Ny, Nx), dtype=bool)

    # Build design matrix: (N_volumes, 7)
    B = build_design_matrix(bvals, bvecs)

    # Get masked voxel signals: (N_pixels, N_volumes)
    pixels = dwi_signal[mask]

    # Clamp to avoid log(0)
    pixels = np.maximum(pixels, 1e-10)
    log_signal = np.log(pixels)  # (N_pixels, N_volumes)

    # Solve: log_signal^T = B @ params => params = pinv(B) @ log_signal^T
    # params shape: (7, N_pixels)
    params, _, _, _ = np.linalg.lstsq(B, log_signal.T, rcond=None)

    # Extract results
    ln_S0 = params[0]              # (N_pixels,)
    tensor_flat = params[1:7]      # (6, N_pixels)

    S0_map[mask] = np.exp(ln_S0)
    tensor_elems[mask] = tensor_flat.T  # (N_pixels, 6)

    # Clamp S0 and tensor elements to physical range
    S0_map = np.clip(S0_map, 0, None)
    S0_map = np.where(np.isfinite(S0_map), S0_map, 0.0)
    tensor_elems = np.where(np.isfinite(tensor_elems), tensor_elems, 0.0)

    return tensor_elems, S0_map


def fit_dti_wls(dwi_signal, bvals, bvecs, mask=None):
    """Estimate diffusion tensors via weighted least squares.

    Two-step procedure:
    1. OLS fit to get initial parameter estimates
    2. WLS fit using weights = exp(2 * B @ ols_params), which accounts
       for the heteroscedasticity introduced by the log transform

    Parameters
    ----------
    dwi_signal : np.ndarray
        DWI signal, shape (Ny, Nx, N_volumes).
    bvals : np.ndarray
        b-values in s/mm^2, shape (N_volumes,).
    bvecs : np.ndarray
        Gradient directions, shape (N_volumes, 3).
    mask : np.ndarray or None
        Boolean tissue mask, shape (Ny, Nx). If None, all pixels are fit.

    Returns
    -------
    tensor_elems : np.ndarray
        Fitted tensor elements [Dxx,Dxy,Dxz,Dyy,Dyz,Dzz], shape (Ny, Nx, 6).
    S0_map : np.ndarray
        Estimated S0, shape (Ny, Nx).
    """
    bvals = np.asarray(bvals, dtype=np.float64)
    bvecs = np.asarray(bvecs, dtype=np.float64)
    dwi_signal = np.asarray(dwi_signal, dtype=np.float64)
    Ny, Nx, N_volumes = dwi_signal.shape

    tensor_elems = np.zeros((Ny, Nx, 6), dtype=np.float64)
    S0_map = np.zeros((Ny, Nx), dtype=np.float64)

    if mask is None:
        mask = np.ones((Ny, Nx), dtype=bool)

    # Build design matrix: (N_volumes, 7)
    B = build_design_matrix(bvals, bvecs)

    # Get masked voxel signals: (N_pixels, N_volumes)
    pixels = dwi_signal[mask]
    N_pixels = pixels.shape[0]

    # Clamp to avoid log(0)
    pixels = np.maximum(pixels, 1e-10)
    log_signal = np.log(pixels)  # (N_pixels, N_volumes)

    # Step 1: OLS fit
    ols_params, _, _, _ = np.linalg.lstsq(B, log_signal.T, rcond=None)
    # ols_params shape: (7, N_pixels)

    # Step 2: WLS fit
    # Predicted log signal from OLS: (N_volumes, N_pixels) = B @ ols_params
    predicted_log_s = B @ ols_params  # (N_volumes, N_pixels)
    # Weights: w_i = S_i^2 = exp(2 * predicted_log_s_i)
    weights = np.exp(2.0 * predicted_log_s)  # (N_volumes, N_pixels)

    # Clamp weights to avoid overflow
    weights = np.minimum(weights, 1e10)
    weights = np.maximum(weights, 1e-10)

    # Solve weighted least squares per voxel
    # For each voxel: minimize sum_i w_i * (log_signal_i - B_i @ params)^2
    # Solution: (B^T W B)^{-1} B^T W y
    wls_params = np.zeros((7, N_pixels), dtype=np.float64)
    for j in range(N_pixels):
        W = np.diag(weights[:, j])
        BtWB = B.T @ W @ B
        BtWy = B.T @ W @ log_signal[j]
        try:
            wls_params[:, j] = np.linalg.solve(BtWB, BtWy)
        except np.linalg.LinAlgError:
            wls_params[:, j] = ols_params[:, j]

    # Extract results
    ln_S0 = wls_params[0]
    tensor_flat = wls_params[1:7]

    S0_map[mask] = np.exp(ln_S0)
    tensor_elems[mask] = tensor_flat.T

    # Clamp to physical range
    S0_map = np.clip(S0_map, 0, None)
    S0_map = np.where(np.isfinite(S0_map), S0_map, 0.0)
    tensor_elems = np.where(np.isfinite(tensor_elems), tensor_elems, 0.0)

    return tensor_elems, S0_map


def tensor_eig_decomposition(tensor_elems, mask=None):
    """Eigendecompose fitted tensors to get FA, MD, and eigenvectors.

    Parameters
    ----------
    tensor_elems : np.ndarray
        Tensor elements [Dxx,Dxy,Dxz,Dyy,Dyz,Dzz], shape (Ny, Nx, 6).
    mask : np.ndarray or None
        Boolean tissue mask, shape (Ny, Nx).

    Returns
    -------
    eigenvalues : np.ndarray
        Sorted eigenvalues (descending), shape (Ny, Nx, 3).
    eigenvectors : np.ndarray
        Corresponding eigenvectors, shape (Ny, Nx, 3, 3).
        eigenvectors[..., :, i] is the i-th eigenvector.
    fa_map : np.ndarray
        Fractional anisotropy, shape (Ny, Nx).
    md_map : np.ndarray
        Mean diffusivity, shape (Ny, Nx).
    """
    Ny, Nx = tensor_elems.shape[:2]
    eigenvalues = np.zeros((Ny, Nx, 3), dtype=np.float64)
    eigenvectors = np.zeros((Ny, Nx, 3, 3), dtype=np.float64)

    if mask is None:
        mask = np.ones((Ny, Nx), dtype=bool)

    ys, xs = np.where(mask)
    for idx in range(len(ys)):
        y, x = ys[idx], xs[idx]
        elems = tensor_elems[y, x]  # (6,)
        D = tensor_from_elements(
            elems[0], elems[1], elems[2],
            elems[3], elems[4], elems[5],
        )
        evals, evecs = np.linalg.eigh(D)
        # Sort descending
        sort_idx = np.argsort(evals)[::-1]
        evals = evals[sort_idx]
        evecs = evecs[:, sort_idx]
        # Clamp negative eigenvalues to small positive
        evals = np.maximum(evals, 0.0)
        eigenvalues[y, x] = evals
        eigenvectors[y, x] = evecs

    fa_map = compute_fa(eigenvalues)
    fa_map = np.where(mask, fa_map, 0.0)
    md_map = compute_md(eigenvalues)
    md_map = np.where(mask, md_map, 0.0)

    return eigenvalues, eigenvectors, fa_map, md_map
