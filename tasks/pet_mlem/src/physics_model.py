"""
PET emission tomography forward model.

In 2D PET, the system matrix A maps an activity distribution (image)
to a sinogram of line-of-response (LOR) counts. For parallel-ring 2D PET,
this is equivalent to a Radon transform.

The measurement model is Poisson:
    y_i ~ Poisson(sum_j A_ij x_j + r_i)

where y is the measured sinogram, x is the activity image, A is the
system matrix, and r is the background (randoms + scatter).

Uses scikit-image's Radon transform as the system matrix operator.
"""

import numpy as np
from skimage.transform import radon, iradon


def pet_forward_project(image, theta):
    """Forward project activity image to sinogram.

    Computes line integrals of the image along each angle (Radon transform),
    serving as the PET system matrix operation A*x.

    Parameters
    ----------
    image : np.ndarray
        Activity distribution, shape (N, N).
    theta : np.ndarray
        Projection angles in degrees, shape (n_angles,).

    Returns
    -------
    sinogram : np.ndarray
        Projected sinogram, shape (n_radial, n_angles).
        n_radial is determined by scikit-image based on image size.
    """
    sinogram = radon(image, theta=theta, circle=True)
    sinogram = np.maximum(sinogram, 0.0)
    return sinogram


def pet_back_project(sinogram, theta, N):
    """Back-project sinogram to image space (adjoint of forward projection).

    Uses unfiltered back-projection (iradon with no filter), which
    computes A^T * y.

    Parameters
    ----------
    sinogram : np.ndarray
        Sinogram, shape (n_radial, n_angles).
    theta : np.ndarray
        Projection angles in degrees, shape (n_angles,).
    N : int
        Output image size.

    Returns
    -------
    image : np.ndarray
        Back-projected image, shape (N, N).
    """
    image = iradon(sinogram, theta=theta, output_size=N,
                   filter_name=None, circle=True)
    return image


def compute_sensitivity_image(theta, N):
    """Compute the sensitivity image A^T * 1.

    This is the back-projection of a uniform sinogram (all ones),
    representing the total sensitivity of the scanner at each pixel.
    Used as the normalization denominator in MLEM.

    Parameters
    ----------
    theta : np.ndarray
        Projection angles in degrees.
    N : int
        Image size.

    Returns
    -------
    sensitivity : np.ndarray
        Sensitivity image, shape (N, N). Values > 0 inside FOV.
    """
    # Forward project a dummy image to get sinogram dimensions
    dummy = np.ones((N, N))
    sino_shape = radon(dummy, theta=theta, circle=True).shape
    ones_sino = np.ones(sino_shape, dtype=np.float64)
    sensitivity = pet_back_project(ones_sino, theta, N)
    return sensitivity


def add_poisson_noise(sinogram, scale=1.0, rng=None):
    """Add Poisson noise to sinogram.

    Simulates photon counting statistics in PET.

    Parameters
    ----------
    sinogram : np.ndarray
        Expected (clean) sinogram counts.
    scale : float
        Count level scaling. Higher = more counts = lower relative noise.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    noisy : np.ndarray
        Poisson-noisy sinogram (integer counts cast to float).
    """
    if rng is None:
        rng = np.random.default_rng()

    expected = np.maximum(sinogram * scale, 0.0)
    noisy = rng.poisson(expected).astype(np.float64)
    return noisy


def add_background(sinogram, randoms_fraction=0.1, rng=None):
    """Add uniform random coincidence background to sinogram.

    In real PET, random coincidences contribute a roughly uniform
    background to the sinogram.

    Parameters
    ----------
    sinogram : np.ndarray
        Clean sinogram.
    randoms_fraction : float
        Fraction of mean true counts to add as randoms.
    rng : np.random.Generator or None

    Returns
    -------
    sinogram_with_bg : np.ndarray
        Sinogram with added background.
    background : np.ndarray
        The background level (constant array, same shape as sinogram).
    """
    if rng is None:
        rng = np.random.default_rng()

    mean_counts = np.mean(sinogram[sinogram > 0]) if np.any(sinogram > 0) else 1.0
    bg_level = randoms_fraction * mean_counts
    background = np.full_like(sinogram, bg_level)
    sinogram_with_bg = sinogram + background
    return sinogram_with_bg, background
