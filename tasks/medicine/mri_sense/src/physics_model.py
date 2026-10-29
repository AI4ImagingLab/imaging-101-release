"""
Multi-Coil Cartesian MRI Forward Model for SENSE
===================================================

Implements the SENSE encoding model:

    y_c = M * F(S_c * x)    for each coil c

where x is the single-coil image, S_c are coil sensitivity maps,
F is the centered 2D DFT, and M is the undersampling mask.

The key difference from GRAPPA: SENSE operates in image domain by
solving the encoding equation directly, rather than interpolating
missing k-space samples.

The encoding operator A maps image x to multi-coil undersampled k-space:
    A: (Nx*Ny,) -> (Nx*Ny*Nc,)
    A(x) = mask * FFT(x * S)

The adjoint A^H maps k-space back to image:
    A^H: (Nx*Ny*Nc,) -> (Nx*Ny,)
    A^H(y) = sum_c conj(S_c) * IFFT(y_c)

Reference
---------
Pruessmann et al., "SENSE: Sensitivity Encoding for Fast MRI,"
MRM 42.5 (1999): 952-962.
"""

import numpy as np


def centered_fft2(x, axes=(0, 1)):
    """Centered 2D FFT."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes
    )


def centered_ifft2(x, axes=(0, 1)):
    """Centered 2D IFFT."""
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(x, axes=axes), axes=axes), axes=axes
    )


def sense_forward(x, sens, mask):
    """
    SENSE forward operator: image → undersampled multi-coil k-space.

    Parameters
    ----------
    x : (Nx, Ny) complex128
        Single-coil image.
    sens : (Nx, Ny, Nc) complex128
        Coil sensitivity maps.
    mask : (Nx, Ny) or (Nx,) bool
        Undersampling mask.

    Returns
    -------
    y : (Nx, Ny, Nc) complex128
        Undersampled multi-coil k-space.
    """
    # Expand 1D mask to 2D if needed
    if mask.ndim == 1:
        mask_2d = mask[:, None]
    else:
        mask_2d = mask
    return centered_fft2(x[..., None] * sens) * mask_2d[..., None]


def sense_adjoint(y, sens):
    """
    SENSE adjoint operator: multi-coil k-space → image.

    Parameters
    ----------
    y : (Nx, Ny, Nc) complex128
        Multi-coil k-space data.
    sens : (Nx, Ny, Nc) complex128
        Coil sensitivity maps.

    Returns
    -------
    x : (Nx, Ny) complex128
        Combined image estimate.
    """
    return np.sum(sens.conj() * centered_ifft2(y), axis=-1)


def sos_combine(kspace):
    """Root sum-of-squares coil combination from k-space."""
    imspace = centered_ifft2(kspace)
    return np.sqrt(np.sum(np.abs(imspace) ** 2, axis=-1))


def zero_filled_recon(kspace_us):
    """Zero-filled reconstruction: IFFT + RSS."""
    return sos_combine(kspace_us)
