"""Preprocessing: load raw InSAR data and extract wrapped phase and gradients."""

import json
import numpy as np


def load_data(data_path):
    """Load raw interferogram data from .npz file.

    Parameters
    ----------
    data_path : str
        Path to raw_data.npz.

    Returns
    -------
    dict with keys: interferogram, wrapped_phase, magnitude,
        snaphu_unwrapped_phase, snaphu_magnitude.
    """
    raw = np.load(data_path)
    # Arrays are stored with a leading batch dimension (1, H, W); squeeze it.
    return {k: raw[k][0] for k in raw.files}


def load_metadata(meta_path):
    """Load imaging metadata from JSON file.

    Parameters
    ----------
    meta_path : str
        Path to meta_data.json.

    Returns
    -------
    dict
    """
    with open(meta_path) as f:
        return json.load(f)


def extract_phase_and_coherence(interferogram):
    """Extract wrapped phase and coherence from complex interferogram.

    Parameters
    ----------
    interferogram : ndarray, complex64
        Complex interferogram (rows x columns).

    Returns
    -------
    wrapped_phase : ndarray, float32
        Phase angle in [-pi, pi].
    coherence : ndarray, float32
        Normalized magnitude (proxy for coherence), range [0, 1].
    """
    wrapped_phase = np.angle(interferogram).astype(np.float32)
    magnitude = np.abs(interferogram).astype(np.float32)
    mag_max = magnitude.max()
    coherence = magnitude / mag_max if mag_max > 0 else magnitude
    return wrapped_phase, coherence


def est_wrapped_gradient(arr, dtype="float32"):
    """Estimate wrapped gradient of a phase image.

    Computes finite differences and wraps them to [-pi, pi]
    using the standard adjustment (Eq. 2-3 in the paper).

    Parameters
    ----------
    arr : ndarray
        2D wrapped phase image.
    dtype : str
        Output dtype.

    Returns
    -------
    phi_x : ndarray
        Wrapped x-gradient (forward difference along columns).
    phi_y : ndarray
        Wrapped y-gradient (forward difference along rows).
    """
    arr = arr.astype(dtype)
    # Forward difference along columns (x)
    phi_x = np.concatenate([
        arr[:, 1:] - arr[:, :-1],
        np.zeros((arr.shape[0], 1), dtype=arr.dtype)
    ], axis=1)
    # Forward difference along rows (y)
    phi_y = np.concatenate([
        arr[1:, :] - arr[:-1, :],
        np.zeros((1, arr.shape[1]), dtype=arr.dtype)
    ], axis=0)
    # Wrap to [-pi, pi]
    phi_x = np.where(np.abs(phi_x) > np.pi,
                     phi_x - 2 * np.pi * np.sign(phi_x), phi_x)
    phi_y = np.where(np.abs(phi_y) > np.pi,
                     phi_y - 2 * np.pi * np.sign(phi_y), phi_y)
    return phi_x, phi_y
