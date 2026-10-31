"""
Synthetic Measurement Generation for Reflection-Mode ODT
=========================================================

Generates simulated intensity measurements by running the reflection
BPM forward model on the synthetic phantom for all illumination angles.
"""

import os
import numpy as np
import torch

from .preprocessing import prepare_data
from .physics_model import ReflectionBPMConfig, ReflectionBPMForwardModel


def generate_measurements(data_dir: str = "data",
                          device: str = "cpu") -> tuple:
    """
    Generate synthetic reflection-mode ODT measurements.

    Loads the phantom and metadata, builds the forward model, and
    simulates intensity images for all illumination angles.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.
    device : str
        PyTorch device ('cpu' or 'cuda').

    Returns
    -------
    measurements : ndarray, shape (n_angles, Ny, Nx)
        Simulated intensity images.
    phantom_dn : ndarray, shape (Nz, Ny, Nx)
        Ground truth RI contrast volume.
    metadata : dict
        Imaging parameters.
    """
    phantom_dn, metadata = prepare_data(data_dir)
    config = ReflectionBPMConfig.from_metadata(metadata)
    model = ReflectionBPMForwardModel(config, device=device)

    dn_tensor = torch.tensor(phantom_dn, dtype=torch.float64, device=device)

    print(f"  Simulating {config.n_angles} angles...")
    with torch.no_grad():
        meas_tensor = model.forward(dn_tensor)

    measurements = meas_tensor.cpu().numpy()
    print(f"  Intensity range: [{measurements.min():.6f}, {measurements.max():.6f}]")

    return measurements, phantom_dn, metadata


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    measurements, phantom_dn, metadata = generate_measurements(device=device)

    np.savez("data/raw_data.npz", measurements=measurements[np.newaxis])
    np.savez("data/ground_truth.npz", delta_n=phantom_dn[np.newaxis])
    print(f"Saved data/raw_data.npz     measurements: {measurements[np.newaxis].shape}")
    print(f"Saved data/ground_truth.npz delta_n:      {phantom_dn[np.newaxis].shape}")
