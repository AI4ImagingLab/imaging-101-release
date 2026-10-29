"""
Data Loading and Phantom Generation for Reflection-Mode ODT
============================================================

Provides utilities to load imaging metadata and generate a synthetic
4-layer resolution-target phantom for the rMSBP forward model.
"""

import json
import os
import numpy as np


def load_metadata(data_dir: str = "data") -> dict:
    """
    Load imaging metadata from the meta_data JSON file.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing ``meta_data``.

    Returns
    -------
    dict
        Metadata dictionary with imaging parameters.
    """
    meta_path = os.path.join(data_dir, "meta_data.json")
    with open(meta_path) as f:
        return json.load(f)


def generate_phantom(metadata: dict) -> np.ndarray:
    """
    Generate a synthetic 4-layer USAF-like resolution target.

    Each layer contains binary bar patterns at different orientations
    and spatial frequencies, mimicking the resolution targets used in
    the rMS-FPT paper (Zhu et al., 2025).

    Layer layout:
        - Layer 0: vertical bars (3 groups of increasing frequency)
        - Layer 1: horizontal bars (3 groups)
        - Layer 2: diagonal bars (45 degrees)
        - Layer 3: concentric rings

    Parameters
    ----------
    metadata : dict
        Must contain ``volume_shape`` and ``ri_contrast``.

    Returns
    -------
    phantom : ndarray, shape (Nz, Ny, Nx)
        RI contrast volume (Δn).  Background = 0, pattern regions = ri_contrast.
    """
    nz, ny, nx = metadata["volume_shape"]
    dn = metadata["ri_contrast"]

    phantom = np.zeros((nz, ny, nx), dtype=np.float64)

    # Helper: centre coordinates
    y = np.arange(ny) - ny / 2.0
    x = np.arange(nx) - nx / 2.0
    X, Y = np.meshgrid(x, y)

    # ── Layer 0: Vertical bars (3 groups at different frequencies) ──
    layer = np.zeros((ny, nx), dtype=np.float64)
    # Group 1: wide bars (period=20 px), left third
    mask = (X < -nx / 6)
    period = 20
    bars = ((X % period) < period / 2)
    layer[mask & bars] = dn
    # Group 2: medium bars (period=12 px), centre third
    mask = (np.abs(X) <= nx / 6)
    period = 12
    bars = ((X % period) < period / 2)
    layer[mask & bars] = dn
    # Group 3: fine bars (period=6 px), right third
    mask = (X > nx / 6)
    period = 6
    bars = ((X % period) < period / 2)
    layer[mask & bars] = dn
    # Confine to central 80% vertically
    vert_mask = np.abs(Y) < 0.4 * ny
    layer[~vert_mask] = 0.0
    phantom[0] = layer

    # ── Layer 1: Horizontal bars (rotated 90° from Layer 0) ──
    layer = np.zeros((ny, nx), dtype=np.float64)
    # Group 1: wide bars (period=20 px), top third
    mask = (Y < -ny / 6)
    period = 20
    bars = ((Y % period) < period / 2)
    layer[mask & bars] = dn
    # Group 2: medium bars (period=12 px), centre third
    mask = (np.abs(Y) <= ny / 6)
    period = 12
    bars = ((Y % period) < period / 2)
    layer[mask & bars] = dn
    # Group 3: fine bars (period=6 px), bottom third
    mask = (Y > ny / 6)
    period = 6
    bars = ((Y % period) < period / 2)
    layer[mask & bars] = dn
    horiz_mask = np.abs(X) < 0.4 * nx
    layer[~horiz_mask] = 0.0
    phantom[1] = layer

    # ── Layer 2: Diagonal bars (45°) ──
    layer = np.zeros((ny, nx), dtype=np.float64)
    diag = X + Y
    # Group 1 (top-left quadrant): wide
    mask = (X < 0) & (Y < 0)
    period = 16
    bars = ((diag % period) < period / 2)
    layer[mask & bars] = dn
    # Group 2 (bottom-right quadrant): fine
    mask = (X >= 0) & (Y >= 0)
    period = 8
    bars = ((diag % period) < period / 2)
    layer[mask & bars] = dn
    # Confine within a circle of radius 0.35*nx
    R = np.sqrt(X**2 + Y**2)
    layer[R > 0.35 * nx] = 0.0
    phantom[2] = layer

    # ── Layer 3: Concentric rings ──
    layer = np.zeros((ny, nx), dtype=np.float64)
    R = np.sqrt(X**2 + Y**2)
    # Ring 1: r=8..12
    layer[(R >= 8) & (R < 12)] = dn
    # Ring 2: r=18..22
    layer[(R >= 18) & (R < 22)] = dn
    # Ring 3: r=30..33
    layer[(R >= 30) & (R < 33)] = dn
    # Ring 4: r=40..42
    layer[(R >= 40) & (R < 42)] = dn
    # Disk in centre: r<4
    layer[R < 4] = dn
    phantom[3] = layer

    return phantom


def prepare_data(data_dir: str = "data") -> tuple:
    """
    Load metadata and generate the synthetic phantom.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing ``meta_data``.

    Returns
    -------
    phantom_dn : ndarray, shape (Nz, Ny, Nx)
        RI contrast volume.
    metadata : dict
        Imaging parameters.
    """
    metadata = load_metadata(data_dir)
    phantom_dn = generate_phantom(metadata)

    print(f"  Phantom shape : {phantom_dn.shape}")
    print(f"  Δn range      : [{phantom_dn.min():.4f}, {phantom_dn.max():.4f}]")
    print(f"  Non-zero voxels: {np.count_nonzero(phantom_dn)} / {phantom_dn.size}")

    return phantom_dn, metadata
