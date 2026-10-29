"""
Synthetic Measurement Generation for SSNP-ODT
===============================================

Loads the phantom TIFF and simulates ODT complex field measurements
using the SSNP forward model. Unlike IDT, the full complex field
(amplitude and phase) is retained — no abs() operation is applied.
"""

import os
import json
import numpy as np
import torch
import tifffile

from .physics_model import SSNPConfig, SSNPForwardModel


def _load_phantom(data_dir: str) -> tuple:
    """Load phantom TIFF and metadata, return (phantom_dn, metadata)."""
    meta_path = os.path.join(data_dir, "meta_data.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    tiff_path = os.path.join(data_dir, "sample.tiff")
    try:
        raw = tifffile.imread(tiff_path).astype(np.float64)
    except ValueError as exc:
        msg = str(exc)
        if "requires the 'imagecodecs' package" in msg:
            raise RuntimeError(
                "Reading sample.tiff requires 'imagecodecs'. "
                "Install with `pip install imagecodecs` and rerun."
            ) from exc
        raise

    phantom_dn = raw * (metadata["tiff_scale"] / 65535.0) * metadata["ri_contrast_scale"]
    return phantom_dn, metadata


def generate_measurements(data_dir: str = "data",
                          device: str = "cpu") -> tuple:
    """
    Load phantom and simulate ODT complex field measurements via SSNP.

    The forward model propagates a tilted plane wave through the 3D RI
    volume and returns the complex exit field at the detector — retaining
    both amplitude and phase (no squared modulus).

    Parameters
    ----------
    data_dir : str
        Path to data directory containing sample.tiff and meta_data.json.
    device : str
        PyTorch device ('cpu' or 'cuda').

    Returns
    -------
    measurements : ndarray, shape (n_angles, Ny, Nx), complex128
        Complex field images for each illumination angle.
    phantom_dn : ndarray, shape (Nz, Ny, Nx), float64
        Ground truth RI contrast volume.
    metadata : dict
        Imaging parameters.
    """
    phantom_dn, metadata = _load_phantom(data_dir)
    config = SSNPConfig.from_metadata(metadata)

    print(f"  Phantom shape : {phantom_dn.shape}")
    print(f"  Δn range      : [{phantom_dn.min():.6f}, {phantom_dn.max():.6f}]")
    print(f"  Config res    : {config.res}")

    model = SSNPForwardModel(config, device=device)
    dn_tensor = torch.tensor(phantom_dn, dtype=torch.float64, device=device)

    print(f"  Simulating {config.n_angles} illumination angles (complex fields)...")
    with torch.no_grad():
        fields = model.forward(dn_tensor)  # (n_angles, Ny, Nx) complex128

    measurements = fields.cpu().numpy()  # complex128, no abs()
    amp = np.abs(measurements)
    print(f"  Field amplitude range: [{amp.min():.6f}, {amp.max():.6f}]")

    return measurements, phantom_dn, metadata


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    measurements, phantom_dn, metadata = generate_measurements(device=device)

    # Save complex measurements and ground truth
    np.savez("data/raw_data.npz", measurements=measurements[np.newaxis])
    np.savez("data/ground_truth.npz", delta_n=phantom_dn[np.newaxis].astype(np.float32))
    print(f"Saved data/raw_data.npz     measurements: {measurements[np.newaxis].shape}  dtype: {measurements.dtype}")
    print(f"Saved data/ground_truth.npz delta_n:      {phantom_dn[np.newaxis].shape} (float32)")
