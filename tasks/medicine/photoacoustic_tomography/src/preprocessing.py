"""
Data loading and preprocessing for photoacoustic tomography.
"""

import json
import numpy as np


def load_raw_data(data_dir="data"):
    """Load raw PA signals and detector positions from npz files.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    signals : np.ndarray, shape (n_time, n_det_x, n_det_y)
        Recorded pressure time-series (batch dim removed).
    detector_x : np.ndarray, shape (n_det_x,)
        X-coordinates of detectors in meters.
    detector_y : np.ndarray, shape (n_det_y,)
        Y-coordinates of detectors in meters.
    time_vector : np.ndarray, shape (n_time,)
        Time samples in seconds.
    """
    raw = np.load(f"{data_dir}/raw_data.npz")
    signals = raw["signals"][0]       # remove batch dim
    detector_x = raw["detector_x"][0]
    detector_y = raw["detector_y"][0]
    time_vector = raw["time_vector"][0]
    return signals, detector_x, detector_y, time_vector


def load_ground_truth(data_dir="data"):
    """Load ground truth target image.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    gt_image : np.ndarray, shape (nx, ny)
        Binary ground truth image.
    image_x : np.ndarray, shape (nx,)
        X-coordinates of image pixels in meters.
    image_y : np.ndarray, shape (ny,)
        Y-coordinates of image pixels in meters.
    """
    gt = np.load(f"{data_dir}/ground_truth.npz")
    gt_image = gt["ground_truth_image"][0]  # remove batch dim
    image_x = gt["image_x"][0]
    image_y = gt["image_y"][0]
    return gt_image, image_x, image_y


def load_metadata(data_dir="data"):
    """Load imaging parameters from JSON.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    meta : dict
        Dictionary of imaging parameters.
    """
    with open(f"{data_dir}/meta_data.json", "r") as f:
        meta = json.load(f)
    return meta
