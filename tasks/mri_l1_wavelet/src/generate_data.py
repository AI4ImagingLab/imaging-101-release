"""
Synthetic Multi-Coil MRI Data Generation
=========================================

Generates synthetic multi-coil MRI data using:
- Shepp-Logan phantom (128x128) as the ground truth image
- Gaussian coil sensitivity maps (8 coils)
- Random phase-encode undersampling mask (8x acceleration)

The data is fully self-contained without requiring external datasets.
"""

import os
import json
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from physics_model import fft2c, generate_undersampling_mask


def generate_gaussian_csm(
    n_coils: int,
    image_shape: tuple,
    sigma: float = 0.4,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate Gaussian coil sensitivity maps.

    Each coil has a Gaussian sensitivity profile centered at a different
    position around the image, simulating a receive coil array.

    Parameters
    ----------
    n_coils : int
        Number of coils.
    image_shape : tuple (H, W)
        Spatial dimensions.
    sigma : float
        Standard deviation of the Gaussian (relative to image size).
    seed : int
        Random seed for coil center positions.

    Returns
    -------
    csm : ndarray, (n_coils, H, W) complex128
        Normalized coil sensitivity maps.
    """
    rng = np.random.RandomState(seed)
    H, W = image_shape
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float64)
    y_grid = y_grid / H - 0.5
    x_grid = x_grid / W - 0.5

    # Place coil centers evenly around a circle
    angles = np.linspace(0, 2 * np.pi, n_coils, endpoint=False)
    radius = 0.3
    centers_y = radius * np.sin(angles)
    centers_x = radius * np.cos(angles)

    csm = np.zeros((n_coils, H, W), dtype=np.complex128)
    for c in range(n_coils):
        dy = y_grid - centers_y[c]
        dx = x_grid - centers_x[c]
        magnitude = np.exp(-(dy ** 2 + dx ** 2) / (2 * sigma ** 2))
        # Add a smooth phase variation
        phase = 2 * np.pi * rng.rand() + 0.5 * np.pi * (dy + dx)
        csm[c] = magnitude * np.exp(1j * phase)

    # Normalize so that RSS = 1 everywhere
    rss = np.sqrt(np.sum(np.abs(csm) ** 2, axis=0, keepdims=True))
    rss = np.maximum(rss, 1e-12)
    csm = csm / rss

    return csm


def generate_phantom(image_size: int = 128) -> np.ndarray:
    """
    Generate a Shepp-Logan phantom at the specified resolution.

    Parameters
    ----------
    image_size : int
        Spatial resolution (image_size x image_size).

    Returns
    -------
    phantom : ndarray, (image_size, image_size) complex128
        Complex-valued phantom (real-valued, stored as complex for consistency).
    """
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (image_size, image_size), anti_aliasing=True)
    # Normalize to [0, 1]
    phantom = phantom / phantom.max()
    return phantom.astype(np.complex128)


def generate_data(
    image_size: int = 128,
    n_coils: int = 8,
    acceleration_ratio: int = 8,
    acs_fraction: float = 0.08,
    mask_seed: int = 0,
    csm_seed: int = 42,
    output_dir: str = "data",
):
    """
    Generate complete synthetic multi-coil MRI dataset.

    Creates:
    - data/raw_data.npz: masked_kspace, sensitivity_maps, undersampling_mask
    - data/ground_truth.npz: phantom
    - data/meta_data.json: imaging parameters

    Parameters
    ----------
    image_size : int
        Spatial resolution.
    n_coils : int
        Number of receive coils.
    acceleration_ratio : int
        Undersampling factor.
    acs_fraction : float
        Fraction of ACS lines.
    mask_seed : int
        Seed for mask generation.
    csm_seed : int
        Seed for coil sensitivity map generation.
    output_dir : str
        Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {image_size}x{image_size} Shepp-Logan phantom...")
    phantom = generate_phantom(image_size)

    print(f"Generating {n_coils} Gaussian coil sensitivity maps...")
    csm = generate_gaussian_csm(n_coils, (image_size, image_size), seed=csm_seed)

    print("Computing fully-sampled multi-coil k-space...")
    coil_images = csm * phantom[None, :, :]
    full_kspace = fft2c(coil_images)  # (C, H, W)

    print(f"Generating {acceleration_ratio}x undersampling mask...")
    mask = generate_undersampling_mask(
        total_lines=image_size,
        acceleration_ratio=acceleration_ratio,
        acs_fraction=acs_fraction,
        pattern="random",
        seed=mask_seed,
    )

    masked_kspace = full_kspace * mask[None, None, :]  # (C, H, W)

    # Batch-first convention: (1, ...)
    masked_kspace_batch = masked_kspace[None, ...]  # (1, C, H, W)
    csm_batch = csm[None, ...]  # (1, C, H, W)
    phantom_batch = phantom[None, None, ...]  # (1, 1, H, W)

    n_sampled = int(mask.sum())
    print(f"  Sampled {n_sampled}/{image_size} lines ({100*n_sampled/image_size:.1f}%)")

    # Save raw_data.npz
    raw_path = os.path.join(output_dir, "raw_data.npz")
    np.savez_compressed(
        raw_path,
        masked_kspace=masked_kspace_batch,
        sensitivity_maps=csm_batch,
        undersampling_mask=mask,
    )
    print(f"  Saved {raw_path}")

    # Save ground_truth.npz
    gt_path = os.path.join(output_dir, "ground_truth.npz")
    np.savez_compressed(gt_path, phantom=phantom_batch)
    print(f"  Saved {gt_path}")

    # Save meta_data.json
    meta = {
        "image_size": [image_size, image_size],
        "n_coils": n_coils,
        "n_samples": 1,
        "acceleration_ratio": acceleration_ratio,
        "acs_fraction": acs_fraction,
        "mask_pattern": "random",
        "mask_seed": mask_seed,
        "mask_orientation": "vertical",
        "fft_normalization": "ortho",
        "data_source": "synthetic Shepp-Logan phantom with Gaussian coil maps",
        "sensitivity_estimation": "analytic Gaussian model",
    }
    meta_path = os.path.join(output_dir, "meta_data.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {meta_path}")

    print("Data generation complete.")
    return phantom, csm, masked_kspace, mask


if __name__ == "__main__":
    generate_data()
