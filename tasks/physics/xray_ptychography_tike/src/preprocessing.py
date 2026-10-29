"""Preprocessing utilities for X-ray ptychography reconstruction.

Functions for loading raw data, preparing scan positions, initializing
the object array, and configuring probe modes.
"""

import json
import os

import numpy as np


def load_raw_data(data_dir: str) -> dict:
    """Load raw ptychography data from an .npz archive.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing raw_data.npz.

    Returns
    -------
    dict
        Dictionary with keys 'diffraction_patterns', 'scan_positions',
        and 'probe_guess', each as numpy arrays with the batch dimension
        removed (index 0).
    """
    path = os.path.join(data_dir, "raw_data.npz")
    archive = np.load(path)
    return {
        "diffraction_patterns": archive["diffraction_patterns"][0],
        "scan_positions": archive["scan_positions"][0],
        "probe_guess": archive["probe_guess"][0],
    }


def load_metadata(data_dir: str) -> dict:
    """Load imaging metadata from meta_data.json.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing meta_data.json.

    Returns
    -------
    dict
        Parsed JSON metadata dictionary.
    """
    path = os.path.join(data_dir, "meta_data.json")
    with open(path, "r") as f:
        return json.load(f)


def shift_scan_positions(scan: np.ndarray, offset: float = 20.0) -> np.ndarray:
    """Shift scan positions so the minimum coordinate equals offset.

    This ensures all scan positions map to valid regions within the
    object array, with a small buffer at the boundary.

    Parameters
    ----------
    scan : (N, 2) float32
        Raw scan positions in pixel coordinates.
    offset : float
        The desired minimum coordinate value after shifting.

    Returns
    -------
    scan_shifted : (N, 2) float32
        Shifted scan positions.
    """
    scan_shifted = scan.copy()
    scan_shifted -= np.amin(scan_shifted, axis=-2) - offset
    return scan_shifted


def initialize_psi(scan: np.ndarray, probe_shape: tuple,
                   n_slices: int = 1, buffer: int = 2,
                   fill_value: complex = 0.5 + 0j) -> np.ndarray:
    """Initialize the object array (psi) for ptychographic reconstruction.

    Creates a 3D complex array of shape (D, H, W) where D is the number
    of slices (1 for single-slice ptychography). The spatial dimensions
    are computed to contain all scan positions plus the probe footprint
    plus a small buffer.

    Parameters
    ----------
    scan : (N, 2) float32
        Shifted scan positions (after calling shift_scan_positions).
    probe_shape : tuple
        Shape of the probe array; last two elements give (W, H) of the probe.
    n_slices : int
        Number of depth slices. Use 1 for single-slice ptychography.
    buffer : int
        Extra pixels added beyond the maximum scan extent + probe width.
    fill_value : complex
        Initial complex value for all pixels.

    Returns
    -------
    psi : (D, H, W) complex64
        Initialized object array.
    """
    scan_max = np.amax(scan, axis=-2)
    probe_w = probe_shape[-2]
    probe_h = probe_shape[-1]
    psi_shape = (
        n_slices,
        int(np.ceil(scan_max[0])) + probe_w + buffer,
        int(np.ceil(scan_max[1])) + probe_h + buffer,
    )
    psi = np.full(psi_shape, dtype=np.complex64,
                  fill_value=np.complex64(fill_value))
    return psi


def add_probe_modes(probe: np.ndarray, n_modes: int = 1) -> np.ndarray:
    """Add incoherent probe modes via random phase initialization.

    New modes are created by taking the amplitude of the first mode
    and applying a random phase. If n_modes equals the current number
    of modes, the probe is returned unchanged.

    Parameters
    ----------
    probe : (..., M, W, H) complex64
        Probe array with M existing modes.
    n_modes : int
        Desired total number of modes.

    Returns
    -------
    probe : (..., n_modes, W, H) complex64
        Probe with the requested number of modes.
    """
    current_modes = probe.shape[-3]
    if n_modes <= current_modes:
        return probe

    # Create new modes with random phase and same amplitude as mode 0
    amplitude = np.abs(probe[..., 0:1, :, :])  # (..., 1, W, H)
    new_modes = []
    for _ in range(n_modes - current_modes):
        random_phase = np.exp(
            2j * np.pi * np.random.rand(*amplitude.shape).astype(np.float32)
        )
        new_mode = (amplitude * random_phase).astype(np.complex64)
        new_modes.append(new_mode)

    return np.concatenate([probe] + new_modes, axis=-3)
