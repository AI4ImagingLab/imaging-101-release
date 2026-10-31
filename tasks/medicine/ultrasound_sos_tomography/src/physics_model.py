"""
Physics model for ultrasound speed-of-sound tomography.

The forward model computes travel times through a 2D slowness field using
straight-ray (Radon) geometry. In ultrasound transmission tomography, a ring
of transducers surrounds the object. Each transmitter emits a pulse that
travels through the tissue to each receiver. The travel time along a straight
ray path is the line integral of slowness s(r) = 1/c(r):

    t_ij = integral_{ray_ij} s(r) dl  =  A_ij . s

where A is the system matrix (ray-pixel intersection lengths). For parallel-beam
geometry, this is exactly the Radon transform of the slowness field.

We use scikit-image's radon/iradon as the forward/adjoint operators, treating
the slowness field as the image and the sinogram as the set of travel times.
"""

import numpy as np
from skimage.transform import radon, iradon


def radon_forward(slowness, angles_deg):
    """Compute travel-time sinogram from a 2D slowness field.

    This is the forward model for straight-ray transmission tomography:
    each sinogram value is the line integral of slowness along a ray.

    Parameters
    ----------
    slowness : np.ndarray, shape (H, W)
        2D slowness field in s/m. Should be square (H == W).
    angles_deg : np.ndarray, shape (n_angles,)
        Projection angles in degrees.

    Returns
    -------
    sinogram : np.ndarray, shape (n_detectors, n_angles)
        Travel-time sinogram. Each value is the integrated slowness
        (travel time in seconds, scaled by pixel size).
    """
    sinogram = radon(slowness, theta=angles_deg, circle=True)
    return sinogram


def filtered_back_projection(sinogram, angles_deg, output_size=None,
                              filter_name="ramp"):
    """Reconstruct a slowness field from a travel-time sinogram using FBP.

    Parameters
    ----------
    sinogram : np.ndarray, shape (n_detectors, n_angles)
        Travel-time sinogram.
    angles_deg : np.ndarray, shape (n_angles,)
        Projection angles in degrees.
    output_size : int or None
        Size of the output image. If None, inferred from sinogram.
    filter_name : str
        Filter to use: 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'.

    Returns
    -------
    slowness_recon : np.ndarray, shape (output_size, output_size)
        Reconstructed slowness field.
    """
    reconstruction = iradon(sinogram, theta=angles_deg, output_size=output_size,
                            filter_name=filter_name, circle=True)
    return reconstruction


def adjoint_projection(sinogram, angles_deg, output_size):
    """Back-project a sinogram without filtering (adjoint of Radon transform).

    Parameters
    ----------
    sinogram : np.ndarray, shape (n_detectors, n_angles)
        Sinogram data.
    angles_deg : np.ndarray, shape (n_angles,)
        Projection angles in degrees.
    output_size : int
        Size of the output image.

    Returns
    -------
    backproj : np.ndarray, shape (output_size, output_size)
        Backprojected image (unfiltered).
    """
    backproj = iradon(sinogram, theta=angles_deg, output_size=output_size,
                      filter_name=None, circle=True)
    # Scale to approximate the adjoint operator
    backproj *= np.pi / (2 * len(angles_deg))
    return backproj


def add_gaussian_noise(sinogram, noise_std, rng=None):
    """Add Gaussian noise to a travel-time sinogram.

    Parameters
    ----------
    sinogram : np.ndarray
        Clean travel-time sinogram.
    noise_std : float
        Standard deviation of Gaussian noise relative to sinogram max.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    noisy_sinogram : np.ndarray
        Sinogram with added noise.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    noise_level = noise_std * np.max(np.abs(sinogram))
    noise = rng.normal(0, noise_level, size=sinogram.shape)
    return sinogram + noise
