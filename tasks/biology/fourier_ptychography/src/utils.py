"""
Standalone utility functions extracted from PtyLab for Fourier ptychography.

These implementations are functionally identical to PtyLab's originals.
Extracting them here removes the ptylab package dependency from this task.
"""

import numpy as np
from scipy.ndimage import fourier_gaussian


# ---------------------------------------------------------------------------
# FFT utilities
# ---------------------------------------------------------------------------

def fft2c(field):
    """
    Centered unitary 2D FFT (energy-preserving).

    Equivalent to PtyLab's fft2c with fftshiftSwitch=False:
        fftshift(fft2(ifftshift(field), norm='ortho'))
    """
    axes = (-2, -1)
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(field, axes=axes), norm="ortho"), axes=axes
    )


def ifft2c(field):
    """
    Centered unitary 2D IFFT (energy-preserving).

    Equivalent to PtyLab's ifft2c with fftshiftSwitch=False:
        fftshift(ifft2(ifftshift(field), norm='ortho'))
    """
    axes = (-2, -1)
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(field, axes=axes), norm="ortho"), axes=axes
    )


# ---------------------------------------------------------------------------
# Aperture / window functions
# ---------------------------------------------------------------------------

def circ(x, y, D):
    """
    Binary circular aperture on a 2D grid.

    Parameters
    ----------
    x, y : ndarray (2D)
        Coordinate grids (e.g. from np.meshgrid).
    D : float
        Diameter of the circle.

    Returns
    -------
    mask : ndarray, bool
        True inside the circle of diameter D.
    """
    return (x ** 2 + y ** 2) < (D / 2) ** 2


def gaussian2D(n, std):
    """Normalized 2D Gaussian kernel."""
    n = (n - 1) // 2
    x, y = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    h = np.exp(-(x ** 2 + y ** 2) / (2 * std ** 2))
    mask = h < np.finfo(float).eps * np.max(h)
    h *= 1 - mask
    sumh = np.sum(h)
    if sumh != 0:
        h = h / sumh
    return h


# ---------------------------------------------------------------------------
# Probe regularization
# ---------------------------------------------------------------------------

def smooth_amplitude(field, width, aleph, amplitude_only=True):
    """
    Smooth the amplitude of a complex field using a Gaussian low-pass filter.

    Blends smoothed amplitude with the original:
        result = aleph * smoothed + (1 - aleph) * field

    Parameters
    ----------
    field : ndarray, complex
        Input field (2D or higher).
    width : float
        Gaussian sigma [pixels] for scipy.ndimage.fourier_gaussian.
    aleph : float
        Blend weight (0 = no smoothing, 1 = full smoothing).
    amplitude_only : bool
        If True, smooth only the amplitude; preserve the phase.
    """
    gimmel = 1e-5
    if amplitude_only:
        ph_field = field / (np.abs(field) + gimmel)
        A_field = np.abs(field)
    else:
        ph_field = 1
        A_field = field
    F_field = np.fft.fft2(A_field)
    for ax in [-2, -1]:
        F_field = fourier_gaussian(F_field, width, axis=ax)
    field_smooth = np.fft.ifft2(F_field)
    if amplitude_only:
        field_smooth = np.abs(field_smooth) * ph_field
    return aleph * field_smooth + (1 - aleph) * field
