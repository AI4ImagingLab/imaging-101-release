"""
Preprocessing: data loading and initial guess creation.
"""

import json
import numpy as np


def load_raw_data(path):
    """Load raw projection data from an npz file.

    Parameters
    ----------
    path : str
        Path to raw_data.npz.

    Returns
    -------
    dict
        Keys: 'projections' (1, n_angles, n, n) complex64,
              'theta' (1, n_angles) float32.
    """
    data = np.load(path)
    return {
        'projections': data['projections'].astype(np.complex64),
        'theta': data['theta'].astype(np.float32),
    }


def load_ground_truth(path):
    """Load ground truth volume from an npz file.

    Parameters
    ----------
    path : str
        Path to ground_truth.npz.

    Returns
    -------
    dict
        Keys: 'volume' (1, nz, n, n) complex64.
    """
    data = np.load(path)
    return {
        'volume': data['volume'].astype(np.complex64),
    }


def load_metadata(path):
    """Load imaging metadata from a JSON file.

    Parameters
    ----------
    path : str
        Path to meta_data.json.

    Returns
    -------
    dict
        Imaging parameters including volume_shape, n_angles, tilt_rad, etc.
    """
    with open(path, 'r') as f:
        return json.load(f)


def create_initial_guess(volume_shape, dtype=np.complex64):
    """Create a zero-valued initial guess for reconstruction.

    Parameters
    ----------
    volume_shape : tuple of int
        Shape of the 3D volume (nz, n, n).
    dtype : numpy dtype
        Data type of the volume. Default: complex64.

    Returns
    -------
    numpy.ndarray
        Zero-initialized volume of the given shape and dtype.
    """
    return np.zeros(volume_shape, dtype=dtype)
