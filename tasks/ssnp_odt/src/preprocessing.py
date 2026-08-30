"""
Data Preprocessing for SSNP-ODT
=================================

Handles loading complex field measurements and metadata.

Pipeline: raw_data.npz (complex fields) + meta_data.json → measurements + parameters
"""

import os
import json
import numpy as np


def load_measurements(data_dir: str = "data") -> np.ndarray:
    """
    Load complex field measurements from raw_data.npz.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    measurements : ndarray, shape (n_angles, Ny, Nx), complex128
        Complex field images for each illumination angle.
    """
    path = os.path.join(data_dir, "raw_data.npz")
    data = np.load(path)
    return data["measurements"][0].astype(np.complex128)


def load_metadata(data_dir: str = "data") -> dict:
    """
    Load imaging metadata from JSON file.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    dict with keys:
        'volume_shape'      : list[int]   — [Nz, Ny, Nx]
        'res_um'            : list[float]  — voxel size in μm [dx, dy, dz]
        'wavelength_um'     : float        — illumination wavelength in μm
        'n0'                : float        — background refractive index
        'NA'                : float        — objective numerical aperture
        'n_angles'          : int          — number of illumination angles
        'ri_contrast_scale' : float        — RI contrast scaling factor
        'tiff_scale'        : float        — TIFF normalisation factor
    """
    path = os.path.join(data_dir, "meta_data.json")
    with open(path, "r") as f:
        return json.load(f)


def prepare_data(data_dir: str = "data") -> tuple:
    """
    Load measurements and metadata for the SSNP-ODT pipeline.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    measurements : ndarray, shape (n_angles, Ny, Nx), complex128
    metadata : dict
    """
    measurements = load_measurements(data_dir)
    metadata = load_metadata(data_dir)
    return measurements, metadata
