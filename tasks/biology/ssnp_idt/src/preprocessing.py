"""
Data Preprocessing for SSNP-IDT
=================================

Handles loading the phantom TIFF volume and metadata, preparing inputs
for the SSNP forward model and reconstruction.

Pipeline: sample.tiff + meta_data (JSON) → RI contrast volume + parameters
"""

import os
import json
import numpy as np
import tifffile


def load_observation(data_dir: str = "data") -> np.ndarray:
    """
    Load the phantom TIFF and convert to RI contrast Δn.

    The TIFF contains uint16 values. Conversion:
        Δn = tiff_data * (tiff_scale / 65535) * ri_contrast_scale

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing sample.tiff and meta_data.

    Returns
    -------
    phantom_dn : ndarray, shape (Nz, Ny, Nx), float64
        RI contrast volume (Δn = n - n0).
    """
    meta = load_metadata(data_dir)

    tiff_path = os.path.join(data_dir, "sample.tiff")
    try:
        raw = tifffile.imread(tiff_path).astype(np.float64)
    except ValueError as exc:
        msg = str(exc)
        if "requires the 'imagecodecs' package" in msg:
            raise RuntimeError(
                "Reading sample.tiff requires the optional 'imagecodecs' "
                "dependency because the file uses LZW compression. "
                "Install it with `python -m pip install imagecodecs` and rerun."
            ) from exc
        raise

    tiff_scale = meta["tiff_scale"]
    ri_scale = meta["ri_contrast_scale"]
    phantom_dn = raw * (tiff_scale / 65535.0) * ri_scale

    return phantom_dn


def load_metadata(data_dir: str = "data") -> dict:
    """
    Load imaging metadata from JSON file.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing the meta_data file.

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
    Load and prepare all data needed for the SSNP-IDT pipeline.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    phantom_dn : ndarray, shape (Nz, Ny, Nx), float64
        RI contrast volume.
    metadata : dict
        Imaging parameters.
    """
    phantom_dn = load_observation(data_dir)
    metadata = load_metadata(data_dir)
    return phantom_dn, metadata
