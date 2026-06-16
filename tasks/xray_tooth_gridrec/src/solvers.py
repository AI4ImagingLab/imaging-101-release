"""Inverse solvers for parallel-beam X-ray CT reconstruction.

Implements Filtered Back-Projection (FBP) — the standard analytical
reconstruction algorithm for parallel-beam computed tomography.

The gridrec algorithm (Dowd et al., 1999) is an FFT-based variant of
FBP that uses gridding to interpolate from polar to Cartesian coordinates
in Fourier space. The standard FBP implemented here applies the ramp
(Ram-Lak) filter in 1-D Fourier domain and then back-projects, which
is mathematically equivalent.
"""

import numpy as np


def ramp_filter(n_detector):
    """Construct the ramp (Ram-Lak) filter in the frequency domain.

    The ramp filter |omega| is the optimal filter for FBP reconstruction
    of parallel-beam CT data (exact inverse of the Radon transform in
    the continuous limit).

    Parameters
    ----------
    n_detector : int
        Number of detector pixels.

    Returns
    -------
    filt : ndarray, shape (n_fft,)
        Ramp filter in frequency domain, suitable for multiplication
        with the FFT of each projection row.
    """
    n_fft = max(64, int(2 ** np.ceil(np.log2(2 * n_detector))))
    freqs = np.fft.fftfreq(n_fft)
    filt = np.abs(freqs)
    return filt


def filter_sinogram(sinogram, filt=None):
    """Apply the ramp filter to sinogram rows in the Fourier domain.

    Parameters
    ----------
    sinogram : ndarray, shape (n_angles, n_detector)
        Preprocessed sinogram data.
    filt : ndarray, optional
        Frequency-domain filter. If None, uses the ramp filter.

    Returns
    -------
    filtered : ndarray, shape (n_angles, n_detector)
        Filtered sinogram (ready for back-projection).
    """
    n_angles, n_det = sinogram.shape
    n_fft = max(64, int(2 ** np.ceil(np.log2(2 * n_det))))

    if filt is None:
        filt = ramp_filter(n_det)

    filtered = np.zeros_like(sinogram)
    for i in range(n_angles):
        proj_fft = np.fft.fft(sinogram[i], n=n_fft)
        proj_fft *= filt
        proj_filtered = np.real(np.fft.ifft(proj_fft))[:n_det]
        filtered[i] = proj_filtered

    return filtered


def back_project(sinogram, theta, n_pixels):
    """Simple pixel-driven back-projection.

    For each pixel in the reconstruction grid, interpolates the
    sinogram value corresponding to its projected position at each angle.

    Parameters
    ----------
    sinogram : ndarray, shape (n_angles, n_detector)
        Sinogram data (filtered or unfiltered).
    theta : ndarray, shape (n_angles,)
        Projection angles in radians.
    n_pixels : int
        Size of the square reconstruction grid.

    Returns
    -------
    image : ndarray, shape (n_pixels, n_pixels)
        Back-projected image.
    """
    n_det = sinogram.shape[1]
    det_center = (n_det - 1) / 2.0
    img_center = (n_pixels - 1) / 2.0

    image = np.zeros((n_pixels, n_pixels), dtype=np.float64)
    # Row 0 = top of image = positive y in physical coordinates
    y_grid, x_grid = np.mgrid[0:n_pixels, 0:n_pixels]
    x_grid = x_grid.astype(np.float64) - img_center
    # Flip y so that row 0 corresponds to positive y
    y_grid = img_center - y_grid.astype(np.float64)

    for i, angle in enumerate(theta):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        t = x_grid * cos_a + y_grid * sin_a + det_center
        image += np.interp(
            t.ravel(),
            np.arange(n_det, dtype=np.float64),
            sinogram[i],
        ).reshape(n_pixels, n_pixels)

    # Scale by angular step
    if len(theta) > 1:
        dtheta = np.pi / len(theta)
        image *= dtheta
    return image


def filtered_back_projection(sinogram, theta, n_pixels):
    """Filtered Back-Projection (FBP) reconstruction.

    Standard analytical reconstruction for parallel-beam CT:
    1. Apply the ramp filter to each projection in the Fourier domain.
    2. Back-project the filtered sinogram onto the image grid.

    Parameters
    ----------
    sinogram : ndarray, shape (n_angles, n_detector)
        Preprocessed sinogram data (line integrals of attenuation).
    theta : ndarray, shape (n_angles,)
        Projection angles in radians.
    n_pixels : int
        Size of the square reconstruction grid.

    Returns
    -------
    image : ndarray, shape (n_pixels, n_pixels)
        Reconstructed attenuation image.
    """
    filtered = filter_sinogram(sinogram)
    return back_project(filtered, theta, n_pixels)


def circular_mask(image, ratio=0.95):
    """Apply a circular mask to a reconstructed image.

    Pixels outside a circle of radius ``ratio * (n/2)`` centered on
    the image are set to zero. This removes edge artifacts common
    in CT reconstruction.

    Parameters
    ----------
    image : ndarray, shape (n, n)
        Reconstructed image (single slice).
    ratio : float
        Fraction of the half-width used as the mask radius.

    Returns
    -------
    ndarray, shape (n, n)
        Masked image.
    """
    n = image.shape[0]
    center = (n - 1) / 2.0
    y, x = np.ogrid[0:n, 0:n]
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = r <= ratio * n / 2.0
    return image * mask
