"""
Generate synthetic fan-beam CT data.

Creates a Shepp-Logan phantom, computes fan-beam sinograms for both
full-scan and short-scan geometries, and adds Gaussian noise.
"""

import os
import json
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from src.physics_model import (
    fan_beam_geometry,
    fan_beam_forward_vectorized,
    add_gaussian_noise,
)


def create_phantom(N=256):
    """Create a Shepp-Logan phantom.

    Parameters
    ----------
    N : int
        Image size (N x N).

    Returns
    -------
    phantom : np.ndarray
        Phantom image, shape (N, N), values in [0, 1].
    """
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (N, N), anti_aliasing=True, preserve_range=True)
    phantom = phantom.astype(np.float64)
    return phantom


def generate_synthetic_data(
    N=128,
    n_det=192,
    n_angles_full=180,
    n_angles_short=None,
    D_sd=256.0,
    D_dd=256.0,
    sigma_noise=0.02,
    seed=42,
):
    """Generate complete synthetic fan-beam CT dataset.

    Creates both full-scan (360 deg) and short-scan (pi + fan_angle)
    sinograms from a Shepp-Logan phantom.

    Parameters
    ----------
    N : int
        Image size.
    n_det : int
        Number of detector elements.
    n_angles_full : int
        Number of projection angles for full scan.
    n_angles_short : int or None
        Number of angles for short scan. If None, computed from geometry.
    D_sd : float
        Source-to-isocenter distance in pixels.
    D_dd : float
        Isocenter-to-detector distance in pixels.
    sigma_noise : float
        Gaussian noise level (relative to max sinogram value).
    seed : int
        Random seed.

    Returns
    -------
    data : dict
    """
    rng = np.random.default_rng(seed)

    # Create phantom
    phantom = create_phantom(N)

    # Full-scan geometry (2*pi)
    geo_full = fan_beam_geometry(N, n_det, n_angles_full, D_sd, D_dd,
                                 angle_range=2 * np.pi)

    # Compute fan angle for short-scan range
    half_det = geo_full['det_pos'][-1]
    fan_half_angle = np.arctan(half_det / D_sd)
    short_scan_range = np.pi + 2 * fan_half_angle

    if n_angles_short is None:
        # Scale number of angles proportionally
        n_angles_short = int(n_angles_full * short_scan_range / (2 * np.pi))

    geo_short = fan_beam_geometry(N, n_det, n_angles_short, D_sd, D_dd,
                                  angle_range=short_scan_range)

    # Forward project
    print("  Forward projecting (full scan)...")
    sino_full_clean = fan_beam_forward_vectorized(phantom, geo_full)

    print("  Forward projecting (short scan)...")
    sino_short_clean = fan_beam_forward_vectorized(phantom, geo_short)

    # Add noise (relative to max sinogram value)
    abs_sigma = sigma_noise * np.max(sino_full_clean)
    sino_full_noisy = add_gaussian_noise(sino_full_clean, abs_sigma, rng=rng)
    sino_short_noisy = add_gaussian_noise(sino_short_clean, abs_sigma, rng=rng)

    return {
        'phantom': phantom,
        'sino_full_clean': sino_full_clean,
        'sino_full_noisy': sino_full_noisy,
        'sino_short_clean': sino_short_clean,
        'sino_short_noisy': sino_short_noisy,
        'angles_full': geo_full['angles'],
        'angles_short': geo_short['angles'],
        'det_pos': geo_full['det_pos'],
        'N': N,
        'n_det': n_det,
        'n_angles_full': n_angles_full,
        'n_angles_short': n_angles_short,
        'D_sd': D_sd,
        'D_dd': D_dd,
        'sigma_noise': sigma_noise,
        'abs_sigma': abs_sigma,
        'fan_half_angle_deg': np.degrees(fan_half_angle),
        'short_scan_range_deg': np.degrees(short_scan_range),
    }


def save_data(data, task_dir):
    """Save generated data to task data/ directory."""
    data_dir = os.path.join(task_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # raw_data.npz
    np.savez_compressed(
        os.path.join(data_dir, 'raw_data.npz'),
        sino_full=data['sino_full_noisy'][np.newaxis, ...].astype(np.float32),
        sino_short=data['sino_short_noisy'][np.newaxis, ...].astype(np.float32),
        angles_full=data['angles_full'].astype(np.float32),
        angles_short=data['angles_short'].astype(np.float32),
        det_pos=data['det_pos'].astype(np.float32),
    )

    # ground_truth.npz
    np.savez_compressed(
        os.path.join(data_dir, 'ground_truth.npz'),
        phantom=data['phantom'][np.newaxis, ...].astype(np.float32),
    )

    # meta_data.json
    meta = {
        "image_size": int(data['N']),
        "n_det": int(data['n_det']),
        "n_angles_full": int(data['n_angles_full']),
        "n_angles_short": int(data['n_angles_short']),
        "source_to_isocenter_pixels": float(data['D_sd']),
        "isocenter_to_detector_pixels": float(data['D_dd']),
        "noise_sigma_relative": float(data['sigma_noise']),
        "fan_half_angle_deg": float(data['fan_half_angle_deg']),
        "short_scan_range_deg": float(data['short_scan_range_deg']),
        "modality": "fan-beam CT",
        "detector_type": "flat",
    }
    with open(os.path.join(data_dir, 'meta_data.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    task_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Generating synthetic fan-beam CT data...")
    data = generate_synthetic_data()
    save_data(data, task_dir)
    print(f"  phantom: {data['phantom'].shape}")
    print(f"  sino_full: {data['sino_full_noisy'].shape}")
    print(f"  sino_short: {data['sino_short_noisy'].shape}")
    print(f"  fan half-angle: {data['fan_half_angle_deg']:.1f} deg")
    print(f"  short scan range: {data['short_scan_range_deg']:.1f} deg")
