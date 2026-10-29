"""
DCE-MRI Data Preprocessing
===========================

Loads dynamic MRI k-space data, undersampling masks, ground truth,
and metadata from the standard data directory.
"""

import os
import json
import numpy as np


def load_observation(data_dir):
    """
    Load observation data (undersampled k-space and masks).

    Parameters
    ----------
    data_dir : str
        Path to data/ directory.

    Returns
    -------
    obs : dict
        'undersampled_kspace' : ndarray, (T, N, N) complex128
        'undersampling_masks' : ndarray, (T, N, N) float64
    """
    raw = np.load(os.path.join(data_dir, 'raw_data.npz'))
    return {
        'undersampled_kspace': raw['undersampled_kspace'][0].astype(np.complex128),
        'undersampling_masks': raw['undersampling_masks'][0].astype(np.float64),
    }


def load_ground_truth(data_dir):
    """
    Load ground truth dynamic images and time points.

    Parameters
    ----------
    data_dir : str
        Path to data/ directory.

    Returns
    -------
    gt : dict
        'dynamic_images' : ndarray, (T, N, N) float64
        'time_points' : ndarray, (T,) float64
    """
    gt_file = np.load(os.path.join(data_dir, 'ground_truth.npz'))
    return {
        'dynamic_images': gt_file['dynamic_images'][0].astype(np.float64),
        'time_points': gt_file['time_points'].astype(np.float64),
    }


def load_metadata(data_dir):
    """
    Load imaging parameters from meta_data.json.

    Parameters
    ----------
    data_dir : str
        Path to data/ directory.

    Returns
    -------
    meta : dict
        Imaging parameters.
    """
    with open(os.path.join(data_dir, 'meta_data.json'), 'r') as f:
        return json.load(f)


def prepare_data(data_dir):
    """
    Load all data components in one call.

    Parameters
    ----------
    data_dir : str
        Path to data/ directory.

    Returns
    -------
    obs : dict
        Observation data.
    gt : dict
        Ground truth data.
    meta : dict
        Metadata.
    """
    obs = load_observation(data_dir)
    gt = load_ground_truth(data_dir)
    meta = load_metadata(data_dir)
    return obs, gt, meta
