"""
Generate synthetic PET emission tomography data.

Creates a digital phantom with hot lesions, forward-projects it into a
sinogram, and adds Poisson noise to simulate photon counting statistics.
"""

import os
import json
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from src.physics_model import (
    pet_forward_project,
    add_poisson_noise,
    add_background,
)


def create_activity_phantom(N=128):
    """Create a PET activity distribution phantom.

    Uses a modified Shepp-Logan phantom with activity concentrations
    representing different tissue uptake levels. Includes hot lesions
    (high uptake) to simulate tumors.

    Parameters
    ----------
    N : int
        Image size (N x N).

    Returns
    -------
    phantom : np.ndarray
        Activity distribution, shape (N, N), non-negative.
    """
    base = shepp_logan_phantom()
    base = resize(base, (N, N), anti_aliasing=True, preserve_range=True)
    base = base.astype(np.float64)

    # Convert to activity concentrations
    # Background tissue ~ 1.0, hot regions ~ 3-5x background
    activity = np.zeros((N, N), dtype=np.float64)

    # Assign activity levels based on phantom intensity
    activity[base > 0.03] = 1.0    # soft tissue background
    activity[base > 0.15] = 1.5    # moderate uptake
    activity[base > 0.35] = 2.0    # gray matter-like
    activity[base > 0.65] = 3.0    # high uptake region
    activity[base > 0.85] = 0.5    # CSF-like (low activity)

    # Add hot lesions (simulating tumors)
    yy, xx = np.mgrid[:N, :N]
    cx, cy = N / 2, N / 2

    # Lesion 1: small hot spot
    r1 = np.sqrt((xx - cx + 15) ** 2 + (yy - cy - 10) ** 2)
    activity[r1 < 5] = 6.0

    # Lesion 2: larger moderate hot spot
    r2 = np.sqrt((xx - cx - 20) ** 2 + (yy - cy + 8) ** 2)
    activity[r2 < 8] = 4.0

    return activity


def generate_synthetic_data(
    N=128,
    n_angles=120,
    count_level=1000.0,
    randoms_fraction=0.1,
    seed=42,
):
    """Generate complete synthetic PET dataset.

    Parameters
    ----------
    N : int
        Image size.
    n_angles : int
        Number of projection angles.
    count_level : float
        Scale factor for Poisson counts. Higher = more counts = lower noise.
    randoms_fraction : float
        Fraction of true counts to add as random coincidence background.
    seed : int
        Random seed.

    Returns
    -------
    data : dict
    """
    rng = np.random.default_rng(seed)

    # Create phantom
    phantom = create_activity_phantom(N)

    # Projection angles (full ring, 180 degrees sufficient for 2D PET)
    theta = np.linspace(0, 180, n_angles, endpoint=False)

    # Forward project (clean sinogram)
    sino_clean = pet_forward_project(phantom, theta)

    # Add background (randoms)
    sino_with_bg, background = add_background(sino_clean, randoms_fraction, rng=rng)

    # Add Poisson noise
    sino_noisy = add_poisson_noise(sino_with_bg, scale=count_level, rng=rng)

    # Scale back to original count level
    sino_noisy = sino_noisy / count_level
    background_scaled = background  # background is at original scale

    return {
        'phantom': phantom,
        'sino_clean': sino_clean,
        'sino_noisy': sino_noisy,
        'background': background_scaled,
        'theta': theta,
        'N': N,
        'n_angles': n_angles,
        'count_level': count_level,
        'randoms_fraction': randoms_fraction,
    }


def save_data(data, task_dir):
    """Save generated data to task data/ directory."""
    data_dir = os.path.join(task_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(data_dir, 'raw_data.npz'),
        sinogram=data['sino_noisy'][np.newaxis, ...].astype(np.float32),
        background=data['background'][np.newaxis, ...].astype(np.float32),
        theta=data['theta'][np.newaxis, ...].astype(np.float32),
    )

    np.savez_compressed(
        os.path.join(data_dir, 'ground_truth.npz'),
        activity_map=data['phantom'][np.newaxis, ...].astype(np.float32),
    )

    # Determine sinogram shape from clean sinogram
    n_radial = data['sino_clean'].shape[0]

    meta = {
        "image_size": int(data['N']),
        "n_angles": int(data['n_angles']),
        "n_radial_bins": int(n_radial),
        "count_level": float(data['count_level']),
        "randoms_fraction": float(data['randoms_fraction']),
        "noise_model": "poisson",
        "modality": "2D PET",
        "angle_range_deg": 180.0,
    }
    with open(os.path.join(data_dir, 'meta_data.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    task_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Generating synthetic PET data...")
    data = generate_synthetic_data()
    save_data(data, task_dir)
    print(f"  phantom: {data['phantom'].shape}")
    print(f"  sinogram: {data['sino_noisy'].shape}")
    print(f"  count level: {data['count_level']}")
