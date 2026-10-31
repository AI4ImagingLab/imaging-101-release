"""
Synthetic Data Generation for DPI
===================================

Generates a synthetic black hole crescent image and simulated EHT observation
for testing the DPI pipeline without real observation data.

Note: The actual raw_data.npz shipped with this task was created from real
EHT observation files (obs.uvfits, gt.fits) using ehtim >= 1.2 and
astropy >= 4.0. Those extra dependencies are NOT listed in requirements.txt
because the pipeline no longer requires them at runtime. If you need to
re-generate raw_data.npz or ground_truth.npz from the original FITS/UVFITS
files, install them manually:
    pip install ehtim astropy

Reference
---------
Sun & Bouman (2020), arXiv:2010.14462
"""

import os
import json
import numpy as np


def generate_crescent_image(npix: int = 32, fov_uas: float = 160.0,
                             radius_uas: float = 20.0,
                             width_uas: float = 8.0,
                             asymmetry: float = 0.3,
                             total_flux: float = 1.0) -> np.ndarray:
    """
    Generate a synthetic black hole crescent image.

    The crescent is modeled as the difference of two offset Gaussians,
    with Doppler-boosted brightness asymmetry.

    Parameters
    ----------
    npix : int
        Image size in pixels.
    fov_uas : float
        Field of view in microarcseconds.
    radius_uas : float
        Ring radius in microarcseconds.
    width_uas : float
        Ring width (FWHM) in microarcseconds.
    asymmetry : float
        Brightness asymmetry (0 = symmetric, 1 = fully one-sided).
    total_flux : float
        Total flux normalization.

    Returns
    -------
    (npix, npix) ndarray — crescent image
    """
    pixel_size = fov_uas / npix
    center = (npix - 1) / 2.0

    y, x = np.mgrid[0:npix, 0:npix]
    x_uas = (x - center) * pixel_size
    y_uas = (y - center) * pixel_size
    r = np.sqrt(x_uas ** 2 + y_uas ** 2)

    # Gaussian ring
    sigma = width_uas / 2.355
    ring = np.exp(-0.5 * ((r - radius_uas) / sigma) ** 2)

    # Brightness asymmetry (brighter on the south)
    angle = np.arctan2(y_uas, x_uas)
    brightness = 1.0 + asymmetry * np.sin(angle)
    image = ring * brightness

    # Normalize
    if image.sum() > 0:
        image = image * total_flux / image.sum()

    return image


def generate_dataset(npix: int = 32, fov_uas: float = 160.0,
                      data_dir: str = "data") -> None:
    """
    Generate a complete synthetic dataset for DPI testing.

    Creates a crescent image and saves metadata. Note: for full synthetic
    observation generation (UVFITS), use ehtim's observation simulation.

    Parameters
    ----------
    npix : int
        Image size in pixels.
    fov_uas : float
        Field of view in microarcseconds.
    data_dir : str
        Output directory.
    """
    os.makedirs(data_dir, exist_ok=True)

    image = generate_crescent_image(npix=npix, fov_uas=fov_uas)

    np.save(os.path.join(data_dir, "synthetic_gt.npy"), image)

    print(f"Synthetic crescent image saved to {data_dir}/synthetic_gt.npy")
    print(f"  Shape: {image.shape}, Total flux: {image.sum():.4f}")
    print(f"  Peak: {image.max():.6f}")
