"""Preprocessing for MCR: data loading and initial spectra estimation.

Provides SVD-based initial spectral guess from noisy hyperspectral data,
and data loading utilities.
"""

import json
import pathlib

import numpy as np
from scipy.sparse.linalg import svds


def load_observation(data_dir):
    """Load the noisy hyperspectral observation.

    Parameters
    ----------
    data_dir : str or Path
        Path to data/ directory containing raw_data.npz.

    Returns
    -------
    obs : dict
        Keys: 'hsi_noisy' (n_pixels, n_freq), 'wn' (n_freq,).
    """
    data_dir = pathlib.Path(data_dir)
    raw = np.load(data_dir / "raw_data.npz")
    # Strip batch dimension (first axis) from on-disk format
    return {"hsi_noisy": raw["hsi_noisy"][0], "wn": raw["wn"][0]}


def load_ground_truth(data_dir):
    """Load ground truth concentrations and spectra.

    Parameters
    ----------
    data_dir : str or Path
        Path to data/ directory containing ground_truth.npz.

    Returns
    -------
    gt : dict
        Keys: 'concentrations' (M, N, k), 'concentrations_ravel' (M*N, k),
              'spectra' (k, n_freq), 'hsi_clean' (M*N, n_freq).
    """
    data_dir = pathlib.Path(data_dir)
    gt = np.load(data_dir / "ground_truth.npz")
    # Strip batch dimension (first axis) from on-disk format
    return {k: gt[k][0] for k in gt.files}


def load_metadata(data_dir):
    """Load imaging metadata.

    Parameters
    ----------
    data_dir : str or Path
        Path to data/ directory containing meta_data.json.

    Returns
    -------
    meta : dict
    """
    data_dir = pathlib.Path(data_dir)
    with open(data_dir / "meta_data.json") as f:
        return json.load(f)


def estimate_initial_spectra(hsi_noisy, n_components):
    """Estimate initial spectra from SVD of the noisy data.

    Uses truncated SVD to extract the top singular vectors and scales
    them to approximate spectral magnitudes.

    Parameters
    ----------
    hsi_noisy : ndarray, shape (n_pixels, n_freq)
        Noisy hyperspectral data matrix.
    n_components : int
        Number of chemical components.

    Returns
    -------
    initial_spectra : ndarray, shape (n_components, n_freq)
        Non-negative initial spectral estimates.
    """
    U, s, Vh = svds(hsi_noisy, k=n_components + 1)
    # Sort by descending singular value, drop the smallest
    order = np.flip(np.argsort(s))[:-1]
    Vh = Vh[order, :]
    initial_spectra = np.abs(Vh) / Vh.max() * hsi_noisy.max()
    return initial_spectra
