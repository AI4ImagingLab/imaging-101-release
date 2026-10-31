"""
Standalone utility functions extracted from PtyLab for conventional ptychography.

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
    """
    Normalized 2D Gaussian kernel of size (2n-1) × (2n-1).

    Parameters
    ----------
    n : int
        Half-width: kernel spans [-n//2, n//2] on each axis.
    std : float
        Standard deviation in pixels.

    Returns
    -------
    h : ndarray, float64
        Normalized kernel (sums to 1).
    """
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
# Coordinate utilities
# ---------------------------------------------------------------------------

def cart2pol(x, y):
    """
    Cartesian to polar coordinates.

    Returns
    -------
    th : ndarray
        Angle [rad] = arctan2(y, x).
    r : ndarray
        Radius = sqrt(x² + y²).
    """
    th = np.arctan2(y, x)
    r = np.hypot(x, y)
    return th, r


# ---------------------------------------------------------------------------
# Angular spectrum propagator
# ---------------------------------------------------------------------------

def aspw(u, z, wavelength, L, bandlimit=True):
    """
    Angular spectrum plane-wave propagator.

    Propagates the 2D field `u` by distance `z` using the band-limited
    angular spectrum method (Matsushima et al., Opt. Express 2009).

    Parameters
    ----------
    u : ndarray, shape (N, N), complex
        Input field at z = 0.  Assumed to be in the spatial (not Fourier) domain.
    z : float
        Propagation distance [m].
    wavelength : float
        Wavelength [m].
    L : float
        Physical size of the field of view [m] (= N * dx).
    bandlimit : bool
        Apply the Matsushima band-limit to avoid aliasing (default True).

    Returns
    -------
    u_prop : ndarray, shape (N, N), complex
        Propagated field.
    H : ndarray, shape (N, N), complex
        Transfer function (for diagnostics).
    """
    N = u.shape[-1]
    H = _aspw_transfer_function(float(z), float(wavelength), int(N), float(L), bandlimit)
    U = fft2c(u)
    u_prop = ifft2c(U * H)
    return u_prop, H


def _aspw_transfer_function(z, wavelength, N, L, bandlimit=True):
    """Band-limited angular spectrum transfer function."""
    a_z = abs(z)
    k = 2 * np.pi / wavelength
    X = np.arange(-N / 2, N / 2) / L
    Fx, Fy = np.meshgrid(X, X)
    f_max = L / (wavelength * np.sqrt(L ** 2 + 4 * a_z ** 2))
    W = circ(Fx, Fy, 2 * f_max).astype(float)
    exponent = 1 - (Fx * wavelength) ** 2 - (Fy * wavelength) ** 2
    mask = exponent > 0
    if not bandlimit:
        mask = np.ones_like(mask)
    exponent = np.clip(exponent, 0, np.inf)
    H = mask * np.exp(1j * k * a_z * np.sqrt(exponent))
    if z < 0:
        H = H.conj()
    return H * W


# ---------------------------------------------------------------------------
# Fermat spiral scan grid
# ---------------------------------------------------------------------------

def GenerateNonUniformFermat(n, radius=1000, power=1):
    """
    Generate a non-uniform Fermat spiral scan grid.

    Parameters
    ----------
    n : int
        Number of scan points.
    radius : float
        Grid radius (in whatever units you will multiply by dxp).
    power : float
        Spiral power: 1 = standard Fermat, >1 = more points toward center.

    Returns
    -------
    R, C : ndarray, shape (n,)
        Row and column coordinates (in radius units).
    """
    r = np.sqrt(np.arange(0, n) / n)
    theta0 = 137.508 / 180 * np.pi
    theta = np.arange(0, n) * theta0
    C = radius * r ** power * np.cos(theta)
    R = radius * r ** power * np.sin(theta)
    return R, C


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

    Returns
    -------
    result : ndarray, same shape as field
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
