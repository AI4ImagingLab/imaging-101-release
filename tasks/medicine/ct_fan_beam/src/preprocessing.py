"""
Data loading and preprocessing for fan-beam CT.
"""

import os
import json
import numpy as np


def load_sinogram_data(task_dir):
    """Load fan-beam sinogram data from raw_data.npz.

    Parameters
    ----------
    task_dir : str
        Path to the task root directory.

    Returns
    -------
    sino_full : np.ndarray
        Full-scan sinogram, shape (1, n_angles_full, n_det), float64.
    sino_short : np.ndarray
        Short-scan sinogram, shape (1, n_angles_short, n_det), float64.
    angles_full : np.ndarray
        Full-scan angles in radians, shape (n_angles_full,).
    angles_short : np.ndarray
        Short-scan angles in radians, shape (n_angles_short,).
    det_pos : np.ndarray
        Detector element positions, shape (n_det,).
    """
    data_dir = os.path.join(task_dir, 'data')
    raw = np.load(os.path.join(data_dir, 'raw_data.npz'))
    sino_full = raw['sino_full'].astype(np.float64)
    sino_short = raw['sino_short'].astype(np.float64)
    angles_full = raw['angles_full'].astype(np.float64)
    angles_short = raw['angles_short'].astype(np.float64)
    det_pos = raw['det_pos'].astype(np.float64)
    return sino_full, sino_short, angles_full, angles_short, det_pos


def load_ground_truth(task_dir):
    """Load ground truth phantom.

    Returns
    -------
    phantom : np.ndarray
        Ground truth image, shape (1, N, N), float64.
    """
    data_dir = os.path.join(task_dir, 'data')
    gt = np.load(os.path.join(data_dir, 'ground_truth.npz'))
    phantom = gt['phantom'].astype(np.float64)
    return phantom


def load_metadata(task_dir):
    """Load acquisition metadata.

    Returns
    -------
    meta : dict
    """
    data_dir = os.path.join(task_dir, 'data')
    with open(os.path.join(data_dir, 'meta_data.json'), 'r') as f:
        meta = json.load(f)
    return meta


def preprocess_sinogram(sinogram):
    """Preprocess sinogram: remove batch dim.

    Parameters
    ----------
    sinogram : np.ndarray
        Shape (1, n_angles, n_det).

    Returns
    -------
    sino_2d : np.ndarray
        Shape (n_angles, n_det).
    """
    return sinogram[0]
