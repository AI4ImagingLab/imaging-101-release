"""
Data Preprocessing for CG-SENSE MRI Reconstruction
=====================================================

Loads multi-coil k-space data, coil sensitivity maps, and ground
truth phantom. Handles undersampling with ACS region preservation.
"""

import os
import json
import numpy as np


def load_observation(data_dir: str = "data") -> dict:
    """Load multi-coil k-space and sensitivity maps from raw_data.npz."""
    path = os.path.join(data_dir, "raw_data.npz")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_ground_truth(data_dir: str = "data") -> np.ndarray:
    """Load ground truth phantom. Returns (1, Nx, Ny) float32."""
    return np.load(os.path.join(data_dir, "ground_truth.npz"))["image"]


def load_metadata(data_dir: str = "data") -> dict:
    """Load imaging parameters."""
    with open(os.path.join(data_dir, "meta_data.json")) as f:
        return json.load(f)


def get_full_kspace(obs_data: dict) -> np.ndarray:
    """Reconstruct complex k-space. Returns (Nx, Ny, Nc) complex128."""
    real = obs_data["kspace_full_real"][0].astype(np.float64)
    imag = obs_data["kspace_full_imag"][0].astype(np.float64)
    return real + 1j * imag


def get_sensitivity_maps(obs_data: dict) -> np.ndarray:
    """Reconstruct complex sensitivity maps. Returns (Nx, Ny, Nc) complex128."""
    real = obs_data["sensitivity_maps_real"][0].astype(np.float64)
    imag = obs_data["sensitivity_maps_imag"][0].astype(np.float64)
    return real + 1j * imag


def undersample_kspace(kspace_full: np.ndarray, R: int = 4,
                       acs_width: int = 16) -> tuple:
    """
    Undersample k-space in phase-encode direction with ACS preservation.

    Parameters
    ----------
    kspace_full : (Nx, Ny, Nc) complex128
    R : int
        Acceleration factor.
    acs_width : int
        Number of fully-sampled ACS lines at center.

    Returns
    -------
    kspace_us : (Nx, Ny, Nc) complex128
    mask_1d : (Nx,) bool
    """
    Nx = kspace_full.shape[0]
    ctr = Nx // 2
    acs_half = acs_width // 2

    mask_1d = np.zeros(Nx, dtype=bool)
    mask_1d[::R] = True
    mask_1d[ctr - acs_half:ctr + acs_half] = True

    kspace_us = np.zeros_like(kspace_full)
    kspace_us[mask_1d, :, :] = kspace_full[mask_1d, :, :]

    return kspace_us, mask_1d


def prepare_data(data_dir: str = "data", R: int = 4, acs_width: int = 16):
    """
    Load and prepare all data for CG-SENSE reconstruction.

    Returns
    -------
    kspace_us : (Nx, Ny, Nc) complex128
    sens : (Nx, Ny, Nc) complex128
    kspace_full : (Nx, Ny, Nc) complex128
    phantom : (Nx, Ny) float64
    metadata : dict
    """
    obs = load_observation(data_dir)
    gt = load_ground_truth(data_dir)
    meta = load_metadata(data_dir)

    kspace_full = get_full_kspace(obs)
    sens = get_sensitivity_maps(obs)
    phantom = gt[0].astype(np.float64)

    kspace_us, _ = undersample_kspace(kspace_full, R, acs_width)

    return kspace_us, sens, kspace_full, phantom, meta
