"""
GRAPPA Reconstruction Solver
==============================

Implements GeneRalized Autocalibrating Partially Parallel Acquisitions
(GRAPPA) for multi-coil MRI reconstruction.

GRAPPA is a k-space interpolation method that estimates missing k-space
samples from acquired neighbours using linear kernels calibrated from
a fully-sampled auto-calibration signal (ACS) region.

Algorithm:
    1. Extract all overlapping patches from the ACS region
    2. For each unique undersampling geometry (hole pattern):
       a. Collect source samples (acquired neighbours) and target
          samples (center of patch) from all ACS patches
       b. Solve the regularised least-squares problem:
          W = (S^H S + lambda I)^{-1} S^H T
          where S = source matrix, T = target matrix
    3. Apply learned weights to fill holes in the undersampled k-space:
       kspace[hole] = W @ neighbours

The key insight is that linear relationships between k-space samples
are shift-invariant, so weights learned from the ACS region generalise
to the entire k-space.

Reference
---------
Griswold et al., "Generalized Autocalibrating Partially Parallel
Acquisitions (GRAPPA)," MRM 47.6 (2002): 1202-1210.

Implementation adapted from pygrappa (mckib2/pygrappa).
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided

from src.physics_model import centered_ifft2, sos_combine


def _view_as_windows(arr, window_shape):
    """Extract sliding windows from an array (replaces skimage.util.view_as_windows).

    Returns a view of the array with shape (*output_dims, *window_shape),
    where output_dims[i] = arr.shape[i] - window_shape[i] + 1.

    Parameters
    ----------
    arr : np.ndarray
        Input array, must be C-contiguous.
    window_shape : tuple of int
        Shape of the sliding window. Must have same number of dims as arr.

    Returns
    -------
    windows : np.ndarray
        Windowed view (no data copy), shape (*output_dims, *window_shape).
    """
    arr = np.ascontiguousarray(arr)
    window_shape = tuple(window_shape)
    ndim = arr.ndim
    assert len(window_shape) == ndim

    out_shape = tuple(s - w + 1 for s, w in zip(arr.shape, window_shape)) + window_shape
    out_strides = arr.strides * 2  # strides for both output dims and window dims
    return as_strided(arr, shape=out_shape, strides=out_strides)


def grappa_reconstruct(
    kspace_us: np.ndarray,
    calib: np.ndarray,
    kernel_size: tuple = (5, 5),
    lamda: float = 0.01,
) -> np.ndarray:
    """
    Reconstruct missing k-space samples using GRAPPA.

    Parameters
    ----------
    kspace_us : ndarray, (Nx, Ny, Nc) complex128
        Undersampled multi-coil k-space (zeros at missing locations).
    calib : ndarray, (acs_x, Ny, Nc) complex128
        Fully-sampled calibration (ACS) data.
    kernel_size : tuple of int
        GRAPPA kernel size (kx, ky).
    lamda : float
        Tikhonov regularisation parameter.

    Returns
    -------
    kspace_recon : ndarray, (Nx, Ny, Nc) complex128
        Reconstructed k-space with missing entries filled.
    """
    kx, ky = kernel_size
    kx2, ky2 = kx // 2, ky // 2
    nc = calib.shape[-1]
    adjx = kx % 2
    adjy = ky % 2

    # Pad k-space and calibration data
    kspace = np.pad(kspace_us, ((kx2, kx2), (ky2, ky2), (0, 0)), mode="constant")
    calib_pad = np.pad(calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode="constant")

    # Binary mask from first coil (all coils share same pattern)
    mask = np.ascontiguousarray(np.abs(kspace[..., 0]) > 0)

    # Get all overlapping patches from the mask
    P = _view_as_windows(mask, (kx, ky))
    Psh = P.shape[:2]
    P = P.reshape((-1, kx, ky))

    # Find unique sampling patterns
    P, iidx = np.unique(P, return_inverse=True, axis=0)

    # Keep only patterns with a hole at center (these need interpolation)
    validP = np.argwhere(~P[:, kx2, ky2]).squeeze()
    invalidP = np.argwhere(np.all(P == 0, axis=(1, 2)))
    validP = np.setdiff1d(validP, invalidP, assume_unique=True)
    validP = np.atleast_1d(validP)

    # Give P a coil dimension
    P = np.tile(P[..., None], (1, 1, 1, nc))

    # Get all overlapping patches of ACS
    A = _view_as_windows(calib_pad, (kx, ky, nc)).reshape((-1, kx, ky, nc))

    # Initialize reconstruction array
    recon = np.zeros_like(kspace)

    # Train weights and apply for each unique hole geometry
    for ii in validP:
        # Source: acquired samples in ACS patches matching this geometry
        S = A[:, P[ii, ...]]  # (n_patches, n_sources)
        T = A[:, kx2, ky2, :]  # (n_patches, nc) — targets are center samples

        # Tikhonov-regularised least squares: W = (S^H S + lambda I)^-1 S^H T
        ShS = S.conj().T @ S
        ShT = S.conj().T @ T
        lamda0 = lamda * np.linalg.norm(ShS) / ShS.shape[0]
        W = np.linalg.solve(ShS + lamda0 * np.eye(ShS.shape[0]), ShT).T

        # Find all holes matching this geometry and apply weights
        idx = np.unravel_index(np.argwhere(iidx == ii), Psh)
        x, y = np.atleast_1d(idx[0].squeeze()) + kx2, np.atleast_1d(idx[1].squeeze()) + ky2
        for xx, yy in zip(x, y):
            S_hole = kspace[xx - kx2:xx + kx2 + adjx, yy - ky2:yy + ky2 + adjy, :]
            S_hole = S_hole[P[ii, ...]]
            recon[xx, yy, :] = (W @ S_hole[:, None]).squeeze()

    # Combine: fill holes with GRAPPA estimates, keep original data
    result = (recon + kspace)[kx2:-kx2, ky2:-ky2, :]
    return result


def grappa_image_recon(
    kspace_us: np.ndarray,
    calib: np.ndarray,
    kernel_size: tuple = (5, 5),
    lamda: float = 0.01,
) -> np.ndarray:
    """
    Full GRAPPA pipeline: k-space interpolation → IFFT → RSS combine.

    Parameters
    ----------
    kspace_us : (Nx, Ny, Nc) complex128
    calib : (acs_x, Ny, Nc) complex128
    kernel_size : tuple
    lamda : float

    Returns
    -------
    recon : (Nx, Ny) float64
        RSS-combined magnitude image.
    """
    kspace_recon = grappa_reconstruct(kspace_us, calib, kernel_size, lamda)
    imspace = centered_ifft2(kspace_recon)
    return sos_combine(imspace)
