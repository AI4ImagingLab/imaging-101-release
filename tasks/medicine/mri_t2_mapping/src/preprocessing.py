"""
Data loading and preprocessing for MRI T2 mapping.
"""

import os
import json
import numpy as np


def load_multi_echo_data(task_dir):
    """Load multi-echo MRI data from raw_data.npz.

    Parameters
    ----------
    task_dir : str
        Path to the task root directory.

    Returns
    -------
    signal : np.ndarray
        Multi-echo signal, shape (1, Ny, Nx, N_echoes), float64.
    """
    data_dir = os.path.join(task_dir, 'data')
    raw = np.load(os.path.join(data_dir, 'raw_data.npz'))
    signal = raw['multi_echo_signal'].astype(np.float64)
    return signal


def load_ground_truth(task_dir):
    """Load ground truth T2 and M0 maps.

    Parameters
    ----------
    task_dir : str
        Path to the task root directory.

    Returns
    -------
    T2_map : np.ndarray
        Ground truth T2 map in ms, shape (1, Ny, Nx), float64.
    M0_map : np.ndarray
        Ground truth M0 map, shape (1, Ny, Nx), float64.
    tissue_mask : np.ndarray
        Boolean tissue mask, shape (1, Ny, Nx).
    """
    data_dir = os.path.join(task_dir, 'data')
    gt = np.load(os.path.join(data_dir, 'ground_truth.npz'))
    T2_map = gt['T2_map'].astype(np.float64)
    M0_map = gt['M0_map'].astype(np.float64)
    tissue_mask = gt['tissue_mask'].astype(bool)
    return T2_map, M0_map, tissue_mask


def load_metadata(task_dir):
    """Load acquisition metadata.

    Parameters
    ----------
    task_dir : str
        Path to the task root directory.

    Returns
    -------
    meta : dict
        Acquisition parameters including echo_times_ms, noise_sigma, etc.
    """
    data_dir = os.path.join(task_dir, 'data')
    with open(os.path.join(data_dir, 'meta_data.json'), 'r') as f:
        meta = json.load(f)
    meta['echo_times_ms'] = np.array(meta['echo_times_ms'], dtype=np.float64)
    return meta


def preprocess_signal(signal):
    """Preprocess multi-echo signal for T2 fitting.

    Removes batch dimension and ensures non-negative values.

    Parameters
    ----------
    signal : np.ndarray
        Multi-echo signal, shape (1, Ny, Nx, N_echoes).

    Returns
    -------
    signal_2d : np.ndarray
        Signal ready for fitting, shape (Ny, Nx, N_echoes).
    """
    signal_2d = signal[0]  # Remove batch dimension
    signal_2d = np.maximum(signal_2d, 0.0)  # Ensure non-negative
    return signal_2d
