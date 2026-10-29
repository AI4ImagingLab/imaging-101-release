"""Velocity model loading and preprocessing for seismic FWI."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def load_marmousi(path: str, ny: int = 2301, nx: int = 751) -> torch.Tensor:
    """
    Load the Marmousi P-wave velocity model from a raw float32 binary file.

    Args:
        path: Path to marmousi_vp.bin.
        ny: Number of grid points in y (horizontal) direction. Default 2301.
        nx: Number of grid points in x (depth) direction. Default 751.

    Returns:
        v: Velocity tensor of shape (ny, nx), values in m/s.
    """
    v = torch.from_file(path, size=ny * nx, dtype=torch.float32).reshape(ny, nx)
    return v


def preprocess_velocity(
    v_full: torch.Tensor, factor: int = 5
) -> tuple:
    """
    Gaussian-smooth then subsample the velocity model to reduce computation.

    Smoothing (sigma=factor) is applied before subsampling (stride=factor)
    to avoid aliasing. Original spacing is 4m; after subsampling: 4*factor m.

    Args:
        v_full: Full-resolution velocity tensor of shape (ny, nx), in m/s.
        factor: Smoothing sigma and subsampling stride. Default 5.

    Returns:
        v_reduced: Subsampled velocity tensor of shape (ceil(ny/factor), ceil(nx/factor)).
        dx: New grid spacing in meters (4.0 * factor).
    """
    v_np = v_full.detach().cpu().numpy()
    v_smooth = gaussian_filter(v_np, sigma=factor)
    v_reduced_np = v_smooth[::factor, ::factor]
    v_reduced = torch.from_numpy(v_reduced_np.astype(np.float32))
    dx = 4.0 * factor
    return v_reduced, dx


def make_initial_model(v_true: torch.Tensor, sigma: float = 25.0) -> torch.Tensor:
    """
    Construct an initial velocity model by smoothing slowness (1/v).

    Smoothing slowness rather than velocity preserves the correct mean
    wavespeed and avoids over-smoothing slow anomalies.

    Args:
        v_true: True velocity model tensor, shape (ny, nx), in m/s.
        sigma: Gaussian filter sigma in grid points. Default 25.

    Returns:
        v_init: Smoothed initial velocity tensor, same shape as v_true.
    """
    v_np = v_true.detach().cpu().numpy()
    slowness_smooth = gaussian_filter(1.0 / v_np, sigma=sigma)
    v_init_np = (1.0 / slowness_smooth).astype(np.float32)
    return torch.from_numpy(v_init_np)
