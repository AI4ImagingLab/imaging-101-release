"""
Data Preprocessing for PnP-ADMM CS-MRI Reconstruction
=======================================================

Loads brain MRI image (ground truth), k-space undersampling masks,
and measurement noise from raw_data.npz / ground_truth.npz.

All arrays use batch-first convention: (N, ...).
"""

import os
import json
import numpy as np


def load_observation(data_dir: str = "data") -> dict:
    """
    Load CS-MRI observation data from raw_data.npz.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    dict with keys:
        mask_random : ndarray, (1, 256, 256) float32
        mask_radial : ndarray, (1, 256, 256) float32
        mask_cartesian : ndarray, (1, 256, 256) float32
        noises_real : ndarray, (1, 256, 256) float32
        noises_imag : ndarray, (1, 256, 256) float32
    """
    path = os.path.join(data_dir, "raw_data.npz")
    data = np.load(path)
    return {
        "mask_random": data["mask_random"],
        "mask_radial": data["mask_radial"],
        "mask_cartesian": data["mask_cartesian"],
        "noises_real": data["noises_real"],
        "noises_imag": data["noises_imag"],
    }


def load_ground_truth(data_dir: str = "data") -> np.ndarray:
    """
    Load ground truth brain image from ground_truth.npz.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    image : ndarray, (1, 256, 256) float32
        Ground truth image in [0, 1].
    """
    path = os.path.join(data_dir, "ground_truth.npz")
    return np.load(path)["image"]


def load_metadata(data_dir: str = "data") -> dict:
    """
    Load imaging parameters from meta_data.json.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    dict with imaging parameters.
    """
    path = os.path.join(data_dir, "meta_data.json")
    with open(path) as f:
        return json.load(f)


def get_complex_noise(obs_data: dict, scale: float = 3.0) -> np.ndarray:
    """
    Reconstruct complex noise array from real/imag parts and apply scaling.

    Parameters
    ----------
    obs_data : dict
        From load_observation().
    scale : float
        Noise amplitude scaling factor.

    Returns
    -------
    noises : ndarray, (256, 256) complex128
        Scaled complex noise.
    """
    real = obs_data["noises_real"][0].astype(np.float64)
    imag = obs_data["noises_imag"][0].astype(np.float64)
    return (real + 1j * imag) * scale


def get_mask(obs_data: dict, mask_name: str = "random") -> np.ndarray:
    """
    Get a specific undersampling mask.

    Parameters
    ----------
    obs_data : dict
        From load_observation().
    mask_name : str
        One of "random", "radial", "cartesian".

    Returns
    -------
    mask : ndarray, (256, 256) float64
        Binary undersampling mask.
    """
    key = f"mask_{mask_name}"
    return obs_data[key][0].astype(np.float64)


def prepare_data(data_dir: str = "data", mask_name: str = "random"):
    """
    Load and return all data needed for PnP-ADMM reconstruction.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.
    mask_name : str
        Which mask to use.

    Returns
    -------
    im_orig : ndarray, (256, 256) float64
        Ground truth image.
    mask : ndarray, (256, 256) float64
        Undersampling mask.
    noises : ndarray, (256, 256) complex128
        Scaled complex noise.
    metadata : dict
        Imaging parameters.
    """
    obs_data = load_observation(data_dir)
    gt = load_ground_truth(data_dir)
    metadata = load_metadata(data_dir)

    im_orig = gt[0].astype(np.float64)
    mask = get_mask(obs_data, mask_name)
    noises = get_complex_noise(obs_data, scale=metadata["noise_scale"])

    return im_orig, mask, noises, metadata
