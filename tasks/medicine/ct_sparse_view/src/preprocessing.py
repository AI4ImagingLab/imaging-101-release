"""
Preprocessing module for sparse-view CT reconstruction.

Loads phantom, sinogram, and metadata from npz/json files.
"""

import json
import os
import numpy as np


def load_ground_truth(data_dir):
    """Load ground truth phantom image.

    Parameters
    ----------
    data_dir : str
        Path to the data/ directory.

    Returns
    -------
    phantom : np.ndarray, shape (H, W)
        Ground truth phantom image (squeezed from batch dimension).
    """
    gt = np.load(os.path.join(data_dir, "ground_truth.npz"))
    phantom = gt["phantom"].squeeze(0)  # remove batch dim
    return phantom


def load_raw_data(data_dir):
    """Load raw measurement data.

    Parameters
    ----------
    data_dir : str
        Path to the data/ directory.

    Returns
    -------
    data : dict with keys:
        'sinogram_sparse' : np.ndarray, shape (n_angles_sparse, n_detectors)
        'sinogram_full'   : np.ndarray, shape (n_angles_full, n_detectors)
        'angles_sparse'   : np.ndarray, shape (n_angles_sparse,)
        'angles_full'     : np.ndarray, shape (n_angles_full,)
    """
    raw = np.load(os.path.join(data_dir, "raw_data.npz"))
    data = {
        "sinogram_sparse": raw["sinogram_sparse"].squeeze(0),
        "sinogram_full": raw["sinogram_full"].squeeze(0),
        "angles_sparse": raw["angles_sparse"].squeeze(0),
        "angles_full": raw["angles_full"].squeeze(0),
    }
    return data


def load_metadata(data_dir):
    """Load imaging metadata.

    Parameters
    ----------
    data_dir : str
        Path to the data/ directory.

    Returns
    -------
    meta : dict
        Imaging parameters (image_size, n_angles_full, n_angles_sparse, etc.)
    """
    with open(os.path.join(data_dir, "meta_data.json"), "r") as f:
        meta = json.load(f)
    return meta
