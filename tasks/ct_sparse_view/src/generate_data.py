"""
Generate synthetic data for sparse-view CT reconstruction.

Creates a Shepp-Logan phantom, computes full and sparse sinograms,
adds noise, and saves to npz files.
"""

import json
import os
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from .physics_model import radon_transform, add_gaussian_noise


def generate_phantom(image_size=256):
    """Generate Shepp-Logan phantom at given resolution.

    Parameters
    ----------
    image_size : int
        Output image size (image_size x image_size).

    Returns
    -------
    phantom : np.ndarray, shape (image_size, image_size)
    """
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (image_size, image_size), anti_aliasing=True)
    return phantom.astype(np.float64)


def generate_data(task_dir, image_size=256, n_angles_full=180,
                  angle_step_sparse=6, noise_std=0.02, seed=42):
    """Generate and save all data for the sparse-view CT task.

    Parameters
    ----------
    task_dir : str
        Root directory of the task.
    image_size : int
        Phantom size.
    n_angles_full : int
        Number of angles for full sampling.
    angle_step_sparse : int
        Take every Nth angle for sparse view.
    noise_std : float
        Gaussian noise std relative to sinogram max.
    seed : int
        Random seed.
    """
    data_dir = os.path.join(task_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Generate phantom
    phantom = generate_phantom(image_size)

    # Full angles
    angles_full = np.linspace(0, 180, n_angles_full, endpoint=False)

    # Sparse angles (every angle_step_sparse-th angle)
    angles_sparse = angles_full[::angle_step_sparse]
    n_angles_sparse = len(angles_sparse)

    # Compute sinograms
    sinogram_full = radon_transform(phantom, angles_full)
    sinogram_sparse_clean = radon_transform(phantom, angles_sparse)

    # Add noise to sparse sinogram
    sinogram_sparse = add_gaussian_noise(sinogram_sparse_clean, noise_std, rng=rng)

    # Save ground truth (batch-first)
    np.savez(
        os.path.join(data_dir, "ground_truth.npz"),
        phantom=phantom[np.newaxis, ...],  # (1, H, W)
    )

    # Save raw data (batch-first)
    np.savez(
        os.path.join(data_dir, "raw_data.npz"),
        sinogram_sparse=sinogram_sparse[np.newaxis, ...],    # (1, n_det, n_angles_sparse)
        sinogram_full=sinogram_full[np.newaxis, ...],        # (1, n_det, n_angles_full)
        angles_sparse=angles_sparse[np.newaxis, ...],        # (1, n_angles_sparse)
        angles_full=angles_full[np.newaxis, ...],            # (1, n_angles_full)
    )

    # Save metadata (no solver params!)
    meta = {
        "image_size": image_size,
        "n_angles_full": n_angles_full,
        "n_angles_sparse": int(n_angles_sparse),
        "angle_step_sparse": angle_step_sparse,
        "noise_std": noise_std,
        "n_detectors": int(sinogram_full.shape[0]),
        "angle_range_deg": 180.0,
        "pixel_size_mm": 1.0,
        "seed": seed,
    }
    with open(os.path.join(data_dir, "meta_data.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Phantom shape: {phantom.shape}")
    print(f"Full sinogram shape: {sinogram_full.shape}")
    print(f"Sparse sinogram shape: {sinogram_sparse.shape}")
    print(f"Full angles: {n_angles_full}, Sparse angles: {n_angles_sparse}")
    print(f"Data saved to {data_dir}")

    return phantom, sinogram_full, sinogram_sparse, angles_full, angles_sparse


if __name__ == "__main__":
    task_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_data(task_dir)
