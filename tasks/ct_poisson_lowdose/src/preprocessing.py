"""
Data loading and preprocessing for the low-dose CT Poisson task.
"""

import json
import os

import numpy as np


def load_ground_truth(data_dir: str) -> np.ndarray:
    """Load the ground-truth phantom image.

    Args:
        data_dir: Path to the data/ directory.

    Returns:
        2D phantom array of shape (H, W).
    """
    gt = np.load(os.path.join(data_dir, "ground_truth.npz"))
    return gt["phantom"][0]  # strip batch dim -> (H, W)


def load_raw_data(data_dir: str) -> dict:
    """Load raw sinogram data, weights, and angles.

    Args:
        data_dir: Path to the data/ directory.

    Returns:
        Dictionary with keys:
            sinogram_clean      : (num_views, num_channels) clean sinogram
            sinogram_low_dose   : (num_views, num_channels) noisy low-dose sinogram
            sinogram_high_dose  : (num_views, num_channels) noisy high-dose sinogram
            weights_low_dose    : (num_views, num_channels) Poisson weights (low dose)
            weights_high_dose   : (num_views, num_channels) Poisson weights (high dose)
            angles              : (num_views,) projection angles in radians
    """
    raw = np.load(os.path.join(data_dir, "raw_data.npz"))
    return {
        "sinogram_clean": raw["sinogram_clean"][0],        # (V, C)
        "sinogram_low_dose": raw["sinogram_low_dose"][0],  # (V, C)
        "sinogram_high_dose": raw["sinogram_high_dose"][0],
        "weights_low_dose": raw["weights_low_dose"][0],
        "weights_high_dose": raw["weights_high_dose"][0],
        "angles": raw["angles"][0],                         # (V,)
    }


def load_metadata(data_dir: str) -> dict:
    """Load imaging metadata (no solver parameters).

    Args:
        data_dir: Path to the data/ directory.

    Returns:
        Dictionary of imaging parameters.
    """
    with open(os.path.join(data_dir, "meta_data.json")) as f:
        return json.load(f)


def sinogram_to_svmbir(sino_2d: np.ndarray) -> np.ndarray:
    """Convert 2D sinogram (V, C) to SVMBIR's 3D format (V, 1, C).

    SVMBIR expects sinograms with shape (num_views, num_slices, num_channels).
    For 2D reconstruction, num_slices = 1.

    Args:
        sino_2d: Sinogram of shape (num_views, num_channels).

    Returns:
        Sinogram of shape (num_views, 1, num_channels).
    """
    return sino_2d[:, np.newaxis, :]


def weights_to_svmbir(weights_2d: np.ndarray) -> np.ndarray:
    """Convert 2D weights (V, C) to SVMBIR's 3D format (V, 1, C).

    Args:
        weights_2d: Weights of shape (num_views, num_channels).

    Returns:
        Weights of shape (num_views, 1, num_channels).
    """
    return weights_2d[:, np.newaxis, :]
