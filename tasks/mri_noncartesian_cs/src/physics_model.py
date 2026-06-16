"""
Non-Cartesian MRI Forward Model
================================

Implements the multi-coil non-Cartesian MRI forward operator:

    y_c = F_NU(S_c * x) + noise,   c = 1, ..., C

where:
    x     : complex image (H, W)
    S_c   : coil sensitivity map for coil c
    F_NU  : Non-Uniform FFT evaluated at radial k-space locations
    y_c   : measured k-space samples for coil c

The adjoint (gridding reconstruction) applies density compensation and
the adjoint NUFFT, then combines coils:

    x_adj = sum_c conj(S_c) * F_NU^H(w * y_c) / sqrt(sum_c |S_c|^2)

Reference
---------
SigPy: Ong & Lustig, 2019. A Python package for high performance iterative
reconstruction. ISMRM.
"""

import numpy as np
from sigpy import nufft as _sp_nufft, nufft_adjoint as _sp_nufft_adjoint


def nufft_forward(image: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """
    Non-Uniform FFT forward operator.

    Parameters
    ----------
    image : ndarray, (H, W) complex
        Image to transform.
    coord : ndarray, (M, 2) float
        Non-Cartesian k-space coordinates in [-N/2, N/2].

    Returns
    -------
    kdata : ndarray, (M,) complex
        k-space samples at the given coordinates.
    """
    return _sp_nufft(image, coord)


def nufft_adjoint(
    kdata: np.ndarray, coord: np.ndarray, image_shape: tuple
) -> np.ndarray:
    """
    Non-Uniform FFT adjoint operator.

    Parameters
    ----------
    kdata : ndarray, (M,) complex
        k-space samples.
    coord : ndarray, (M, 2) float
        Non-Cartesian k-space coordinates.
    image_shape : tuple
        (H, W) output image shape.

    Returns
    -------
    image : ndarray, (H, W) complex
        Adjoint NUFFT result (gridded image).
    """
    return _sp_nufft_adjoint(kdata, coord, image_shape)


def multicoil_nufft_forward(
    image: np.ndarray, coil_maps: np.ndarray, coord: np.ndarray
) -> np.ndarray:
    """
    Multi-coil non-Cartesian MRI forward operator.

    Parameters
    ----------
    image : ndarray, (H, W) complex
        Complex image to encode.
    coil_maps : ndarray, (C, H, W) complex
        Coil sensitivity maps.
    coord : ndarray, (M, 2) float
        Non-Cartesian k-space coordinates.

    Returns
    -------
    kdata : ndarray, (C, M) complex
        Multi-coil k-space samples.
    """
    n_coils = coil_maps.shape[0]
    n_pts = coord.shape[0]
    kdata = np.zeros((n_coils, n_pts), dtype=np.complex128)
    for c in range(n_coils):
        coil_image = coil_maps[c] * image
        kdata[c] = nufft_forward(coil_image, coord)
    return kdata


def compute_density_compensation(
    coord: np.ndarray, image_shape: tuple, max_iter: int = 30
) -> np.ndarray:
    """
    Compute density compensation weights using pipe iteration.

    The weights compensate for non-uniform sampling density in the
    NUFFT adjoint, enabling the gridding reconstruction to approximate
    the inverse.

    Parameters
    ----------
    coord : ndarray, (M, 2) float
        Non-Cartesian k-space coordinates.
    image_shape : tuple
        (H, W) image dimensions.
    max_iter : int
        Number of iterations for pipe method.

    Returns
    -------
    dcf : ndarray, (M,) float
        Density compensation weights.
    """
    dcf = np.ones(coord.shape[0], dtype=np.float64)
    for _ in range(max_iter):
        # Pipe iteration: w <- w / (F F^H w)
        img_tmp = _sp_nufft_adjoint(dcf, coord, image_shape)
        kdata_tmp = _sp_nufft(img_tmp, coord)
        kdata_tmp_abs = np.abs(kdata_tmp)
        kdata_tmp_abs = np.maximum(kdata_tmp_abs, 1e-12)
        dcf = dcf / kdata_tmp_abs
    return np.abs(dcf)


def gridding_reconstruct(
    kdata: np.ndarray,
    coord: np.ndarray,
    coil_maps: np.ndarray,
    dcf: np.ndarray = None,
) -> np.ndarray:
    """
    Gridding reconstruction (density-compensated adjoint NUFFT + coil combine).

    This is a fast, non-iterative baseline reconstruction.

    Parameters
    ----------
    kdata : ndarray, (C, M) complex
        Multi-coil k-space samples.
    coord : ndarray, (M, 2) float
        Non-Cartesian k-space coordinates.
    coil_maps : ndarray, (C, H, W) complex
        Coil sensitivity maps.
    dcf : ndarray, (M,) float, optional
        Density compensation weights. If None, computed internally.

    Returns
    -------
    image : ndarray, (H, W) complex
        Gridding reconstruction (coil-combined).
    """
    image_shape = coil_maps.shape[1:]
    n_coils = coil_maps.shape[0]

    if dcf is None:
        dcf = compute_density_compensation(coord, image_shape)

    # Apply density compensation and adjoint NUFFT for each coil
    coil_images = np.zeros((n_coils, *image_shape), dtype=np.complex128)
    for c in range(n_coils):
        coil_images[c] = nufft_adjoint(kdata[c] * dcf, coord, image_shape)

    # MVUE coil combination
    combined = np.sum(coil_images * np.conj(coil_maps), axis=0)
    normalization = np.sqrt(np.sum(np.abs(coil_maps) ** 2, axis=0))
    normalization = np.maximum(normalization, 1e-12)
    return combined / normalization
