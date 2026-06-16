"""
Dynamic MRI Forward Model
==========================

Implements the per-frame Cartesian MRI forward and adjoint operators
for dynamic (DCE) imaging with frame-specific undersampling masks.

Forward model per frame:
    y_t = M_t * F(x_t)

where:
    x_t : image at time frame t, (N, N) real or complex
    F   : centered 2D FFT with ortho normalization
    M_t : binary undersampling mask for frame t, (N, N)
    y_t : undersampled k-space for frame t

Adjoint (zero-filled reconstruction):
    x_adj_t = F^H(y_t)

For the full time series:
    Y = A(X)  where  A applies per-frame forward operators
"""

import numpy as np


def fft2c(x):
    """
    Centered 2D FFT with ortho normalization.

    Parameters
    ----------
    x : ndarray, (..., N, N)
        Image-domain data.

    Returns
    -------
    ndarray, (..., N, N) complex
        k-space data.
    """
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'),
        axes=(-2, -1),
    )


def ifft2c(x):
    """
    Centered 2D inverse FFT with ortho normalization.

    Parameters
    ----------
    x : ndarray, (..., N, N)
        k-space data.

    Returns
    -------
    ndarray, (..., N, N) complex
        Image-domain data.
    """
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'),
        axes=(-2, -1),
    )


def forward_single(image, mask):
    """
    Single-frame MRI forward operator.

    Parameters
    ----------
    image : ndarray, (N, N)
        Image at one time frame.
    mask : ndarray, (N, N)
        Binary undersampling mask.

    Returns
    -------
    kspace : ndarray, (N, N) complex
        Undersampled k-space.
    """
    return fft2c(image) * mask


def adjoint_single(kspace, mask=None):
    """
    Single-frame MRI adjoint operator (zero-filled IFFT).

    Parameters
    ----------
    kspace : ndarray, (N, N) complex
        (Undersampled) k-space data.
    mask : ndarray or None
        Not used (included for API symmetry). The mask is already
        applied in the k-space data.

    Returns
    -------
    image : ndarray, (N, N) complex
        Zero-filled reconstruction.
    """
    return ifft2c(kspace)


def forward_dynamic(images, masks):
    """
    Dynamic MRI forward operator applied to all time frames.

    Parameters
    ----------
    images : ndarray, (T, N, N)
        Image time series.
    masks : ndarray, (T, N, N)
        Per-frame undersampling masks.

    Returns
    -------
    kspace : ndarray, (T, N, N) complex
        Undersampled k-space per frame.
    """
    return fft2c(images) * masks


def adjoint_dynamic(kspace, masks=None):
    """
    Dynamic MRI adjoint operator: zero-filled IFFT per frame.

    Parameters
    ----------
    kspace : ndarray, (T, N, N) complex
        Undersampled k-space per frame.
    masks : ndarray or None
        Not used (included for API symmetry).

    Returns
    -------
    images : ndarray, (T, N, N) complex
        Zero-filled reconstructions.
    """
    return ifft2c(kspace)


def normal_operator_dynamic(images, masks):
    """
    Normal operator A^H A for dynamic MRI.

    Applies forward then adjoint: F^H M_t F x_t for each frame.

    Parameters
    ----------
    images : ndarray, (T, N, N)
        Image time series.
    masks : ndarray, (T, N, N)
        Per-frame undersampling masks.

    Returns
    -------
    result : ndarray, (T, N, N) complex
        A^H A x.
    """
    return ifft2c(fft2c(images) * masks)
