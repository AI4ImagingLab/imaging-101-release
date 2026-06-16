"""
Synthetic Measurement Generation for SSNP-IDT
===============================================

Loads the phantom TIFF and simulates IDT intensity measurements
using the SSNP forward model. This replaces the ``forward_model.py``
example from the bu-cisl/SSNP-IDT repository.
"""

import os
import numpy as np
import torch

from .preprocessing import prepare_data
from .physics_model import SSNPConfig, SSNPForwardModel


def generate_measurements(data_dir: str = "data",
                          device: str = "cpu") -> tuple:
    """
    Load phantom and simulate IDT measurements via the SSNP forward model.

    Parameters
    ----------
    data_dir : str
        Path to data directory containing sample.tiff and meta_data.
    device : str
        PyTorch device ('cpu' or 'cuda').

    Returns
    -------
    measurements : ndarray, shape (n_angles, Ny, Nx)
        Simulated intensity images for each illumination angle.
    phantom_dn : ndarray, shape (Nz, Ny, Nx)
        Ground truth RI contrast volume.
    metadata : dict
        Imaging parameters.
    """
    phantom_dn, metadata = prepare_data(data_dir)
    config = SSNPConfig.from_metadata(metadata)

    print(f"  Phantom shape : {phantom_dn.shape}")
    print(f"  Δn range      : [{phantom_dn.min():.6f}, {phantom_dn.max():.6f}]")
    print(f"  Config        : {config.res}")

    model = SSNPForwardModel(config, device=device)

    # Convert phantom to torch tensor
    dn_tensor = torch.tensor(phantom_dn, dtype=torch.float64, device=device)

    print(f"  Simulating {config.n_angles} illumination angles...")
    with torch.no_grad():
        intensities = model.forward(dn_tensor)

    measurements = intensities.cpu().numpy()
    print(f"  Measurement range: [{measurements.min():.6f}, {measurements.max():.6f}]")

    return measurements, phantom_dn, metadata


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    measurements, phantom_dn, metadata = generate_measurements(device=device)

    np.savez("data/raw_data.npz", measurements=measurements[np.newaxis])
    np.savez("data/ground_truth.npz", delta_n=phantom_dn[np.newaxis].astype(np.float32))
    print(f"Saved data/raw_data.npz     measurements: {measurements[np.newaxis].shape}")
    print(f"Saved data/ground_truth.npz delta_n:      {phantom_dn[np.newaxis].shape} (float32)")
