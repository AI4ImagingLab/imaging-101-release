"""
Data Preprocessing for Multi-Coil MRI L1-Wavelet Reconstruction
================================================================

Loads multi-coil MRI data from raw_data.npz (masked k-space, sensitivity maps,
undersampling mask) and ground truth images from ground_truth.npz.

All arrays use batch-first convention: (N, ...).
"""

import os
import json
import numpy as np


def load_observation(data_dir: str = "data") -> dict:
    """
    Load MRI observations from raw_data.npz.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    dict with keys:
        masked_kspace : ndarray, (N, C, H, W) complex128
            Undersampled multi-coil k-space measurements.
        sensitivity_maps : ndarray, (N, C, H, W) complex128
            Coil sensitivity maps (Gaussian model).
        undersampling_mask : ndarray, (W,) float32
            1-D binary mask for phase-encode undersampling.
    """
    path = os.path.join(data_dir, "raw_data.npz")
    data = np.load(path)
    return {
        "masked_kspace": data["masked_kspace"],
        "sensitivity_maps": data["sensitivity_maps"],
        "undersampling_mask": data["undersampling_mask"],
    }


def load_ground_truth(data_dir: str = "data") -> np.ndarray:
    """
    Load ground truth phantom images from ground_truth.npz.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    phantom : ndarray, (N, 1, H, W) complex128
        Ground truth Shepp-Logan phantom images.
    """
    path = os.path.join(data_dir, "ground_truth.npz")
    data = np.load(path)
    return data["phantom"]


def load_metadata(data_dir: str = "data") -> dict:
    """
    Load imaging parameters from meta_data.json.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    dict with imaging parameters (image_size, n_coils, acceleration_ratio, etc.).
    """
    path = os.path.join(data_dir, "meta_data.json")
    with open(path) as f:
        return json.load(f)


def prepare_data(data_dir: str = "data"):
    """
    Load and return all data needed for L1-Wavelet reconstruction.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    obs_data : dict
        Observation data (masked_kspace, sensitivity_maps, undersampling_mask).
    ground_truth : ndarray, (N, 1, H, W) complex128
        Ground truth phantom images.
    metadata : dict
        Imaging parameters.
    """
    obs_data = load_observation(data_dir)
    ground_truth = load_ground_truth(data_dir)
    metadata = load_metadata(data_dir)
    return obs_data, ground_truth, metadata
