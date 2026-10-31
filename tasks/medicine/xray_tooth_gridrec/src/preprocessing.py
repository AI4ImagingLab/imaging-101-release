"""Preprocessing for X-ray transmission CT data.

Handles data loading, flat-field correction, and log-linearization
of raw detector counts into line-integral (sinogram) data.
"""

import json
import numpy as np


def load_observation(data_dir):
    """Load raw observation arrays from data/raw_data.npz.

    Parameters
    ----------
    data_dir : str
        Path to the data/ directory containing raw_data.npz.

    Returns
    -------
    dict with keys:
        projections : ndarray, shape (n_projections, n_sinograms, n_detector_pixels)
        flat_field  : ndarray, shape (n_flat_fields, n_sinograms, n_detector_pixels)
        dark_field  : ndarray, shape (n_dark_fields, n_sinograms, n_detector_pixels)
        theta       : ndarray, shape (n_projections,), projection angles in radians
    """
    raw = np.load(f"{data_dir}/raw_data.npz")
    return {
        "projections": raw["projections"][0],
        "flat_field": raw["flat_field"][0],
        "dark_field": raw["dark_field"][0],
        "theta": raw["theta"][0],
    }


def load_metadata(data_dir):
    """Load imaging parameters from data/meta_data.json.

    Parameters
    ----------
    data_dir : str
        Path to the data/ directory containing meta_data.json.

    Returns
    -------
    dict with imaging parameters (n_projections, n_sinograms,
    n_detector_pixels, theta range, modality).
    """
    with open(f"{data_dir}/meta_data.json") as f:
        return json.load(f)


def normalize(projections, flat_field, dark_field):
    """Flat-field correction of raw projection data.

    Computes (projections - dark) / (flat - dark) where flat and dark
    are averaged over their respective image stacks.

    Parameters
    ----------
    projections : ndarray, shape (n_proj, n_sino, n_det)
        Raw projection images (detector counts).
    flat_field : ndarray, shape (n_flat, n_sino, n_det)
        Flat-field (open beam) images.
    dark_field : ndarray, shape (n_dark, n_sino, n_det)
        Dark-field (no beam) images.

    Returns
    -------
    ndarray, shape (n_proj, n_sino, n_det)
        Normalized transmission images in [0, 1] range (ideally).
    """
    flat_avg = np.mean(flat_field, axis=0, dtype=np.float64)
    dark_avg = np.mean(dark_field, axis=0, dtype=np.float64)
    denom = flat_avg - dark_avg
    denom[denom == 0] = 1.0
    proj = projections.astype(np.float64)
    return (proj - dark_avg) / denom


def minus_log(projections):
    """Compute -log of normalized projections (Beer-Lambert law).

    Converts transmission data T = I/I0 into line integrals of the
    linear attenuation coefficient: -ln(T) = integral(mu, ds).

    Values <= 0 are clipped to a small positive number to avoid
    log(0) or log(negative).

    Parameters
    ----------
    projections : ndarray
        Normalized transmission data (output of ``normalize``).

    Returns
    -------
    ndarray, same shape
        Sinogram data (line integrals of attenuation).
    """
    proj = np.clip(projections, a_min=1e-12, a_max=None)
    return -np.log(proj)
