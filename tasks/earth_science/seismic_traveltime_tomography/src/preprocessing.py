"""Data loading and preprocessing for seismic traveltime tomography."""

import json
import numpy as np
from pathlib import Path


def load_data(data_dir: str = 'data') -> dict:
    """Load raw observations, ground truth, and metadata.

    Args:
        data_dir: Directory containing raw_data.npz, ground_truth.npz,
                  and meta_data.json.

    Returns:
        Dictionary with keys:
            'traveltime_obs': (N_src, N_rec) observed traveltimes (s).
            'sources':        (N_src, 2) source positions [[x, z], ...] (km).
            'receivers':      (N_rec, 2) receiver positions [[x, z], ...] (km).
            'velocity_true':  (Nz, Nx) true velocity model (km/s).
            'meta':           dict of imaging parameters.
    """
    data_dir = Path(data_dir)

    raw = np.load(data_dir / 'raw_data.npz')
    gt  = np.load(data_dir / 'ground_truth.npz')
    with open(data_dir / 'meta_data.json') as f:
        meta = json.load(f)

    return {
        'traveltime_obs': raw['traveltime_obs'][0],    # (N_src, N_rec)
        'sources':        raw['sources'][0],            # (N_src, 2)
        'receivers':      raw['receivers'][0],          # (N_rec, 2)
        'velocity_true':  gt['velocity'][0],            # (Nz, Nx)
        'meta':           meta,
    }


def compute_residuals(T_syn: np.ndarray,
                      T_obs: np.ndarray) -> np.ndarray:
    """Compute traveltime residuals (synthetic minus observed).

    Args:
        T_syn: (N_src, N_rec) synthetic traveltimes (s).
        T_obs: (N_src, N_rec) observed traveltimes (s).

    Returns:
        residuals: (N_src, N_rec) traveltime residuals (s).
    """
    return (T_syn - T_obs).astype(np.float32)


def compute_misfit(residuals: np.ndarray) -> float:
    """Compute the total traveltime misfit χ = Σ residual² / 2.

    Args:
        residuals: (N_src, N_rec) traveltime residuals (s).

    Returns:
        misfit: scalar χ (s²).
    """
    return float(0.5 * np.sum(residuals ** 2))
