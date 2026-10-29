"""
Generate synthetic multi-echo spin-echo MRI data for T2 mapping.

Uses a modified Shepp-Logan phantom with physically realistic T2 and M0
values assigned to each tissue region.
"""

import os
import json
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from src.physics_model import simulate_multi_echo


# Tissue T2 values (ms) and M0 (proton density, arbitrary units)
# Shepp-Logan phantom intensity levels map to tissue types.
# After resize with anti-aliasing, background pixels are at ~0.0 and
# edge pixels may have small positive values. We use a threshold of 0.03
# to separate background from tissue.
#
# Regions are processed in order; later entries override earlier ones.
TISSUE_REGIONS = [
    # (intensity_low, intensity_high, T2_ms, M0, label)
    (0.03, 0.15,    40.0,   0.4,   "bone/scalp"),
    (0.15, 0.35,    70.0,   0.7,   "white matter"),
    (0.35, 0.65,    80.0,   0.8,   "gray matter"),
    (0.65, 0.85,    120.0,  0.9,   "deep gray matter"),
    (0.85, 1.05,    150.0,  1.0,   "CSF/bright tissue"),
]


def create_t2_m0_phantom(N=256):
    """Create T2 and M0 maps from a Shepp-Logan phantom.

    Parameters
    ----------
    N : int
        Image size (N x N).

    Returns
    -------
    T2_map : np.ndarray
        T2 relaxation time map in ms, shape (N, N).
    M0_map : np.ndarray
        Proton density map, shape (N, N).
    tissue_mask : np.ndarray
        Boolean mask of tissue pixels (T2 > 0), shape (N, N).
    """
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (N, N), anti_aliasing=True, preserve_range=True)

    T2_map = np.zeros((N, N), dtype=np.float64)
    M0_map = np.zeros((N, N), dtype=np.float64)

    # Assign tissue parameters based on phantom intensity
    for lo, hi, t2, m0, label in TISSUE_REGIONS:
        region = (phantom >= lo) & (phantom < hi)
        T2_map[region] = t2
        M0_map[region] = m0

    tissue_mask = T2_map > 0

    return T2_map, M0_map, tissue_mask


def generate_synthetic_data(
    N=256,
    echo_times_ms=None,
    sigma=0.02,
    seed=42,
):
    """Generate complete synthetic multi-echo MRI dataset.

    Parameters
    ----------
    N : int
        Image size.
    echo_times_ms : np.ndarray or None
        Echo times in ms. Default: 10 echoes from 10 to 100 ms.
    sigma : float
        Rician noise level.
    seed : int
        Random seed.

    Returns
    -------
    data : dict with keys:
        'multi_echo_signal': shape (1, Ny, Nx, N_echoes)
        'T2_map': shape (1, Ny, Nx)
        'M0_map': shape (1, Ny, Nx)
        'tissue_mask': shape (1, Ny, Nx)
        'echo_times_ms': shape (N_echoes,)
        'sigma': float
    """
    if echo_times_ms is None:
        echo_times_ms = np.arange(10, 110, 10, dtype=np.float64)  # 10..100 ms

    rng = np.random.default_rng(seed)

    T2_map, M0_map, tissue_mask = create_t2_m0_phantom(N)

    # Simulate multi-echo acquisition
    signal = simulate_multi_echo(M0_map, T2_map, echo_times_ms, sigma=sigma, rng=rng)
    # signal shape: (N, N, N_echoes)

    return {
        'multi_echo_signal': signal[np.newaxis, ...],    # (1, N, N, N_echoes)
        'T2_map': T2_map[np.newaxis, ...],               # (1, N, N)
        'M0_map': M0_map[np.newaxis, ...],               # (1, N, N)
        'tissue_mask': tissue_mask[np.newaxis, ...],      # (1, N, N)
        'echo_times_ms': echo_times_ms,
        'sigma': sigma,
    }


def save_data(data, task_dir):
    """Save generated data to the task data/ directory.

    Parameters
    ----------
    data : dict
        Output from generate_synthetic_data.
    task_dir : str
        Path to the task root directory.
    """
    data_dir = os.path.join(task_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # raw_data.npz: multi-echo measurements
    np.savez_compressed(
        os.path.join(data_dir, 'raw_data.npz'),
        multi_echo_signal=data['multi_echo_signal'].astype(np.float32),
    )

    # ground_truth.npz: true T2 and M0 maps
    np.savez_compressed(
        os.path.join(data_dir, 'ground_truth.npz'),
        T2_map=data['T2_map'].astype(np.float32),
        M0_map=data['M0_map'].astype(np.float32),
        tissue_mask=data['tissue_mask'],
    )

    # meta_data.json: acquisition parameters
    meta = {
        "image_size": int(data['T2_map'].shape[1]),
        "n_echoes": int(len(data['echo_times_ms'])),
        "echo_times_ms": data['echo_times_ms'].tolist(),
        "noise_sigma": float(data['sigma']),
        "signal_model": "mono_exponential",
        "modality": "multi-echo spin-echo MRI",
        "field_of_view_mm": 220.0,
        "pixel_size_mm": 220.0 / int(data['T2_map'].shape[1]),
    }
    with open(os.path.join(data_dir, 'meta_data.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    task_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = generate_synthetic_data()
    save_data(data, task_dir)
    print(f"Data saved to {os.path.join(task_dir, 'data')}")
    print(f"  multi_echo_signal: {data['multi_echo_signal'].shape}")
    print(f"  T2_map: {data['T2_map'].shape}")
    print(f"  M0_map: {data['M0_map'].shape}")
    print(f"  echo_times_ms: {data['echo_times_ms']}")
