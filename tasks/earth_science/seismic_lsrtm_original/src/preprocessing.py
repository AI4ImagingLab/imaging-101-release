"""Velocity model loading and preprocessing for seismic LSRTM."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def load_marmousi(path: str, ny: int = 2301, nx: int = 751) -> torch.Tensor:
    """Load the Marmousi P-wave velocity model from a raw float32 binary file.

    Args:
        path: Path to marmousi_vp.bin.
        ny: Number of grid points in y (horizontal). Default 2301.
        nx: Number of grid points in x (depth). Default 751.

    Returns:
        v: Velocity tensor of shape (ny, nx), values in m/s.
    """
    v = torch.from_file(path, size=ny * nx, dtype=torch.float32).reshape(ny, nx)
    return v


def select_subregion(
    v_full: torch.Tensor, ny: int = 600, nx: int = 250
) -> torch.Tensor:
    """Select a subregion of the velocity model for inversion.

    Args:
        v_full: Full velocity model, shape (ny_full, nx_full).
        ny: Number of horizontal grid points to keep. Default 600.
        nx: Number of depth grid points to keep. Default 250.

    Returns:
        v: Subregion velocity tensor, shape (ny, nx).
    """
    return v_full[:ny, :nx].clone()


def make_migration_velocity(v_true: torch.Tensor, sigma: float = 5.0) -> torch.Tensor:
    """Construct a smooth migration velocity by smoothing slowness (1/v).

    Args:
        v_true: True velocity model tensor, shape (ny, nx), in m/s.
        sigma: Gaussian filter sigma in grid points. Default 5.0.

    Returns:
        v_mig: Smoothed migration velocity tensor, same shape as v_true.
    """
    v_np = v_true.detach().cpu().numpy()
    slowness_smooth = gaussian_filter(1.0 / v_np, sigma=sigma)
    v_mig_np = (1.0 / slowness_smooth).astype(np.float32)
    return torch.from_numpy(v_mig_np)
