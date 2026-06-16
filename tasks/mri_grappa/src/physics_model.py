"""
Multi-Coil MRI Forward Model
==============================

Implements the multi-coil Cartesian MRI forward model:

    y_c = M * F(S_c * x)    for each coil c

where:
    x : 2D image (Nx, Ny)
    S_c : coil sensitivity map for coil c
    F : 2D centered DFT (with 1/sqrt(N^2) normalization)
    M : binary k-space undersampling mask (phase-encode lines)
    y_c : measured k-space for coil c

The adjoint (zero-filled reconstruction) combines coils via
root sum-of-squares (RSS):

    x_zf = sqrt(sum_c |F^-1(y_c)|^2)

Reference
---------
Griswold et al., "Generalized Autocalibrating Partially Parallel
Acquisitions (GRAPPA)," MRM 47.6 (2002): 1202-1210.
"""

import numpy as np


def centered_fft2(imspace: np.ndarray) -> np.ndarray:
    """
    Centered 2D FFT with 1/sqrt(N^2) normalization.

    Parameters
    ----------
    imspace : (..., Nx, Ny) or (Nx, Ny, Nc)
        Image-space data. FFT applied over first two spatial dims.

    Returns
    -------
    kspace : same shape, complex128
    """
    N = imspace.shape[0]
    ax = (0, 1)
    return (1 / np.sqrt(N**2)) * np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax
    )


def centered_ifft2(kspace: np.ndarray) -> np.ndarray:
    """
    Centered 2D IFFT (inverse of centered_fft2).

    Parameters
    ----------
    kspace : (Nx, Ny, ...) complex128

    Returns
    -------
    imspace : same shape, complex128
    """
    N = kspace.shape[0]
    ax = (0, 1)
    return np.sqrt(N**2) * np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=ax), axes=ax), axes=ax
    )


def sos_combine(imspace: np.ndarray) -> np.ndarray:
    """
    Root sum-of-squares coil combination.

    Parameters
    ----------
    imspace : (Nx, Ny, Nc) complex128
        Per-coil image-space data.

    Returns
    -------
    combined : (Nx, Ny) float64
        RSS-combined magnitude image.
    """
    return np.sqrt(np.sum(np.abs(imspace) ** 2, axis=-1))


def zero_filled_recon(kspace_us: np.ndarray) -> np.ndarray:
    """
    Zero-filled reconstruction: IFFT of undersampled k-space + RSS combine.

    Parameters
    ----------
    kspace_us : (Nx, Ny, Nc) complex128

    Returns
    -------
    recon : (Nx, Ny) float64
    """
    imspace = centered_ifft2(kspace_us)
    return sos_combine(imspace)


def fully_sampled_recon(kspace_full: np.ndarray) -> np.ndarray:
    """
    Fully-sampled reconstruction: IFFT + RSS combine (reference image).

    Parameters
    ----------
    kspace_full : (Nx, Ny, Nc) complex128

    Returns
    -------
    recon : (Nx, Ny) float64
    """
    imspace = centered_ifft2(kspace_full)
    return sos_combine(imspace)
