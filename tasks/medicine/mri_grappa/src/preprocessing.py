"""
Data Preprocessing for GRAPPA MRI Reconstruction
==================================================

Loads synthetic multi-coil k-space data, coil sensitivity maps,
and ground truth phantom from raw_data.npz / ground_truth.npz.
Handles undersampling and calibration region extraction.

All arrays use batch-first convention: (N, ...).
"""

import os
import json
import numpy as np


def load_observation(data_dir: str = "data") -> dict:
    """
    Load multi-coil k-space data and sensitivity maps from raw_data.npz.

    Returns
    -------
    dict with keys:
        kspace_full_real, kspace_full_imag : (1, Nx, Ny, Nc) float32
        sensitivity_maps_real, sensitivity_maps_imag : (1, Nx, Ny, Nc) float32
    """
    path = os.path.join(data_dir, "raw_data.npz")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_ground_truth(data_dir: str = "data") -> np.ndarray:
    """Load ground truth phantom image. Returns (1, Nx, Ny) float32."""
    path = os.path.join(data_dir, "ground_truth.npz")
    return np.load(path)["image"]


def load_metadata(data_dir: str = "data") -> dict:
    """Load imaging parameters."""
    path = os.path.join(data_dir, "meta_data.json")
    with open(path) as f:
        return json.load(f)


def get_full_kspace(obs_data: dict) -> np.ndarray:
    """Reconstruct complex k-space from real/imag parts. Returns (Nx, Ny, Nc) complex128."""
    real = obs_data["kspace_full_real"][0].astype(np.float64)
    imag = obs_data["kspace_full_imag"][0].astype(np.float64)
    return real + 1j * imag


def get_sensitivity_maps(obs_data: dict) -> np.ndarray:
    """Reconstruct complex sensitivity maps. Returns (Nx, Ny, Nc) complex128."""
    real = obs_data["sensitivity_maps_real"][0].astype(np.float64)
    imag = obs_data["sensitivity_maps_imag"][0].astype(np.float64)
    return real + 1j * imag


def undersample_kspace(kspace_full: np.ndarray, R: int = 2,
                       acs_width: int = 20) -> tuple:
    """
    Undersample k-space in the phase-encode direction (dim 0).

    Parameters
    ----------
    kspace_full : (Nx, Ny, Nc) complex128
    R : int
        Acceleration factor.
    acs_width : int
        Width of fully-sampled auto-calibration signal region.

    Returns
    -------
    kspace_us : (Nx, Ny, Nc) complex128
        Undersampled k-space (zeros at missing lines).
    calib : (acs_width, Ny, Nc) complex128
        Calibration data (fully-sampled ACS region).
    mask : (Nx,) bool
        Phase-encode sampling mask.
    """
    Nx = kspace_full.shape[0]
    ctr = Nx // 2

    mask = np.zeros(Nx, dtype=bool)
    mask[::R] = True
    acs_half = acs_width // 2
    mask[ctr - acs_half:ctr + acs_half] = True

    kspace_us = np.zeros_like(kspace_full)
    kspace_us[mask, :, :] = kspace_full[mask, :, :]

    calib = kspace_full[ctr - acs_half:ctr + acs_half, :, :].copy()

    return kspace_us, calib, mask


def prepare_data(data_dir: str = "data", R: int = 2, acs_width: int = 20):
    """
    Load all data and prepare for GRAPPA reconstruction.

    Returns
    -------
    kspace_us : (Nx, Ny, Nc) complex128
    calib : (acs_width, Ny, Nc) complex128
    kspace_full : (Nx, Ny, Nc) complex128
    phantom : (Nx, Ny) float64
    metadata : dict
    """
    obs = load_observation(data_dir)
    gt = load_ground_truth(data_dir)
    meta = load_metadata(data_dir)

    kspace_full = get_full_kspace(obs)
    phantom = gt[0].astype(np.float64)

    kspace_us, calib, _ = undersample_kspace(kspace_full, R, acs_width)

    return kspace_us, calib, kspace_full, phantom, meta
