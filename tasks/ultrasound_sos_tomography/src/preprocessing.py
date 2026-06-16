"""
Preprocessing module for ultrasound speed-of-sound tomography.

Loads speed-of-sound phantom, travel-time sinograms, and metadata
from npz/json files.
"""

import json
import os
import numpy as np


def load_ground_truth(data_dir):
    """Load ground truth speed-of-sound and slowness phantoms.

    Parameters
    ----------
    data_dir : str
        Path to the data/ directory.

    Returns
    -------
    data : dict with keys:
        'sos_phantom'            : np.ndarray, shape (H, W), speed of sound in m/s
        'slowness_perturbation' : np.ndarray, shape (H, W), slowness perturbation in s/m
    """
    gt = np.load(os.path.join(data_dir, "ground_truth.npz"))
    data = {
        "sos_phantom": gt["sos_phantom"].squeeze(0),
        "slowness_perturbation": gt["slowness_perturbation"].squeeze(0),
    }
    return data


def load_raw_data(data_dir):
    """Load raw measurement data.

    Parameters
    ----------
    data_dir : str
        Path to the data/ directory.

    Returns
    -------
    data : dict with keys:
        'sinogram'       : np.ndarray, shape (n_det, n_angles), noisy travel times
        'sinogram_clean' : np.ndarray, shape (n_det, n_angles), clean travel times
        'sinogram_full'  : np.ndarray, shape (n_det, n_angles_full), full-angle travel times
        'angles'         : np.ndarray, shape (n_angles,), projection angles in degrees
        'angles_full'    : np.ndarray, shape (n_angles_full,), full projection angles
    """
    raw = np.load(os.path.join(data_dir, "raw_data.npz"))
    data = {
        "sinogram": raw["sinogram"].squeeze(0),
        "sinogram_clean": raw["sinogram_clean"].squeeze(0),
        "sinogram_full": raw["sinogram_full"].squeeze(0),
        "angles": raw["angles"].squeeze(0),
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
        Imaging parameters.
    """
    with open(os.path.join(data_dir, "meta_data.json"), "r") as f:
        meta = json.load(f)
    return meta
