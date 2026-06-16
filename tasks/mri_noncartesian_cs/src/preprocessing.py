"""
Data Preprocessing for Non-Cartesian MRI Reconstruction
========================================================

Loads non-Cartesian MRI data from raw_data.npz (multi-coil k-space,
radial trajectory, coil sensitivity maps) and ground truth phantom
from ground_truth.npz.

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
        kdata : ndarray, (N, C, M) complex64
            Multi-coil non-Cartesian k-space measurements.
        coord : ndarray, (N, M, 2) float32
            Non-Cartesian k-space trajectory coordinates.
        coil_maps : ndarray, (N, C, H, W) complex64
            Coil sensitivity maps.
    """
    path = os.path.join(data_dir, "raw_data.npz")
    data = np.load(path)
    return {
        "kdata": data["kdata"],
        "coord": data["coord"],
        "coil_maps": data["coil_maps"],
    }


def load_ground_truth(data_dir: str = "data") -> np.ndarray:
    """
    Load ground truth phantom from ground_truth.npz.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    phantom : ndarray, (N, H, W) complex64
        Ground truth phantom images.
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
    dict with imaging parameters (image_size, n_coils, n_spokes, etc.).
    """
    path = os.path.join(data_dir, "meta_data.json")
    with open(path) as f:
        return json.load(f)


def prepare_data(data_dir: str = "data"):
    """
    Load and return all data needed for non-Cartesian reconstruction.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    obs_data : dict
        Observation data (kdata, coord, coil_maps).
    ground_truth : ndarray, (N, H, W) complex64
        Ground truth phantom images.
    metadata : dict
        Imaging parameters.
    """
    obs_data = load_observation(data_dir)
    ground_truth = load_ground_truth(data_dir)
    metadata = load_metadata(data_dir)
    return obs_data, ground_truth, metadata
