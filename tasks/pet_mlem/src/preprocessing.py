"""
Data loading and preprocessing for PET MLEM reconstruction.
"""

import os
import json
import numpy as np


def load_sinogram_data(task_dir):
    """Load PET sinogram data from raw_data.npz.

    Returns
    -------
    sinogram : np.ndarray
        Noisy sinogram, shape (1, n_radial, n_angles), float64.
    background : np.ndarray
        Background estimate, shape (1, n_radial, n_angles), float64.
    theta : np.ndarray
        Projection angles in degrees, shape (1, n_angles), float64.
    """
    data_dir = os.path.join(task_dir, 'data')
    raw = np.load(os.path.join(data_dir, 'raw_data.npz'))
    sinogram = raw['sinogram'].astype(np.float64)
    background = raw['background'].astype(np.float64)
    theta = raw['theta'].astype(np.float64)
    return sinogram, background, theta


def load_ground_truth(task_dir):
    """Load ground truth activity map.

    Returns
    -------
    activity_map : np.ndarray
        True activity distribution, shape (1, N, N), float64.
    """
    data_dir = os.path.join(task_dir, 'data')
    gt = np.load(os.path.join(data_dir, 'ground_truth.npz'))
    return gt['activity_map'].astype(np.float64)


def load_metadata(task_dir):
    """Load acquisition metadata."""
    data_dir = os.path.join(task_dir, 'data')
    with open(os.path.join(data_dir, 'meta_data.json'), 'r') as f:
        return json.load(f)


def preprocess_sinogram(sinogram):
    """Remove batch dimension and ensure non-negative.

    Parameters
    ----------
    sinogram : np.ndarray, shape (1, n_radial, n_angles)

    Returns
    -------
    sino_2d : np.ndarray, shape (n_radial, n_angles)
    """
    sino_2d = sinogram[0]
    sino_2d = np.maximum(sino_2d, 0.0)
    return sino_2d
