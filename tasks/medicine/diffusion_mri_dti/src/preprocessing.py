"""
Data loading and preprocessing for diffusion MRI DTI.
"""

import os
import json
import numpy as np


def load_dwi_data(task_dir):
    """Load diffusion-weighted MRI data from raw_data.npz.

    Parameters
    ----------
    task_dir : str
        Path to the task root directory.

    Returns
    -------
    dwi_signal : np.ndarray
        DWI signal, shape (1, Ny, Nx, N_volumes), float64.
    bvals : np.ndarray
        b-values in s/mm^2, shape (N_volumes,), float64.
    bvecs : np.ndarray
        Gradient directions, shape (N_volumes, 3), float64.
    """
    data_dir = os.path.join(task_dir, 'data')
    raw = np.load(os.path.join(data_dir, 'raw_data.npz'))
    dwi_signal = raw['dwi_signal'].astype(np.float64)
    bvals = raw['bvals'].astype(np.float64)
    bvecs = raw['bvecs'].astype(np.float64)
    return dwi_signal, bvals, bvecs


def load_ground_truth(task_dir):
    """Load ground truth diffusion tensor maps.

    Parameters
    ----------
    task_dir : str
        Path to the task root directory.

    Returns
    -------
    fa_map : np.ndarray
        Ground truth FA map, shape (1, Ny, Nx), float64.
    md_map : np.ndarray
        Ground truth MD map in mm^2/s, shape (1, Ny, Nx), float64.
    tensor_elements : np.ndarray
        Ground truth tensor elements [Dxx,Dxy,Dxz,Dyy,Dyz,Dzz],
        shape (1, Ny, Nx, 6), float64.
    tissue_mask : np.ndarray
        Boolean tissue mask, shape (1, Ny, Nx).
    """
    data_dir = os.path.join(task_dir, 'data')
    gt = np.load(os.path.join(data_dir, 'ground_truth.npz'))
    fa_map = gt['fa_map'].astype(np.float64)
    md_map = gt['md_map'].astype(np.float64)
    tensor_elements = gt['tensor_elements'].astype(np.float64)
    tissue_mask = gt['tissue_mask'].astype(bool)
    return fa_map, md_map, tensor_elements, tissue_mask


def load_metadata(task_dir):
    """Load acquisition metadata.

    Parameters
    ----------
    task_dir : str
        Path to the task root directory.

    Returns
    -------
    meta : dict
        Acquisition parameters.
    """
    data_dir = os.path.join(task_dir, 'data')
    with open(os.path.join(data_dir, 'meta_data.json'), 'r') as f:
        meta = json.load(f)
    return meta


def preprocess_dwi(dwi_signal, bvals, bvecs):
    """Preprocess DWI data for tensor fitting.

    Removes batch dimension, ensures non-negative values, and separates
    b=0 images from diffusion-weighted images.

    Parameters
    ----------
    dwi_signal : np.ndarray
        DWI signal, shape (1, Ny, Nx, N_volumes).
    bvals : np.ndarray
        b-values, shape (N_volumes,).
    bvecs : np.ndarray
        Gradient directions, shape (N_volumes, 3).

    Returns
    -------
    dwi_2d : np.ndarray
        Signal ready for fitting, shape (Ny, Nx, N_volumes).
    S0 : np.ndarray
        Mean b=0 signal, shape (Ny, Nx).
    """
    dwi_2d = dwi_signal[0]  # Remove batch dimension
    dwi_2d = np.maximum(dwi_2d, 0.0)  # Ensure non-negative

    # Compute mean S0 from b=0 volumes
    b0_mask = bvals < 10  # b-values near 0
    if np.any(b0_mask):
        S0 = np.mean(dwi_2d[..., b0_mask], axis=-1)
    else:
        S0 = np.ones(dwi_2d.shape[:2], dtype=np.float64)

    S0 = np.maximum(S0, 1e-10)  # Avoid division by zero
    return dwi_2d, S0
