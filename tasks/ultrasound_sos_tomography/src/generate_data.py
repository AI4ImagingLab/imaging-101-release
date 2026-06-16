"""
Generate synthetic data for ultrasound speed-of-sound tomography.

Creates a 128x128 speed-of-sound phantom with tissue-like inclusions,
converts to slowness perturbation relative to water background, computes
travel-time sinograms via the Radon transform, adds noise, and saves.

The forward model operates on the slowness perturbation
delta_s = s - s_water = 1/c - 1/c_water, which is zero outside the
object. This avoids artifacts from the circular reconstruction support
of the Radon transform.
"""

import json
import os
import numpy as np

from .physics_model import radon_forward, add_gaussian_noise


def generate_sos_phantom(image_size=128):
    """Generate a speed-of-sound phantom mimicking a breast cross-section.

    The phantom has:
    - Background: water at 1500 m/s
    - Outer ring: fat layer at 1450 m/s
    - Inner tissue: fibroglandular tissue at 1540 m/s
    - Tumor inclusion: 1580 m/s (higher SoS)
    - Small cyst: 1500 m/s (fluid-filled)
    - Small calcification: 2500 m/s (hard inclusion)

    The phantom is masked to be exactly the water background (1500 m/s)
    outside the inscribed circle, matching the Radon transform support.

    Parameters
    ----------
    image_size : int
        Output image size (image_size x image_size).

    Returns
    -------
    sos : np.ndarray, shape (image_size, image_size)
        Speed of sound phantom in m/s.
    """
    sos = np.full((image_size, image_size), 1500.0, dtype=np.float64)

    cx, cy = image_size / 2.0, image_size / 2.0
    y, x = np.ogrid[:image_size, :image_size]

    # Outer boundary: elliptical "breast" region
    rx_outer, ry_outer = 0.40 * image_size, 0.38 * image_size
    outer_mask = ((x - cx) / rx_outer) ** 2 + ((y - cy) / ry_outer) ** 2 <= 1.0

    # Fat layer (ring between outer and inner boundary)
    rx_inner, ry_inner = 0.35 * image_size, 0.33 * image_size
    inner_mask = ((x - cx) / rx_inner) ** 2 + ((y - cy) / ry_inner) ** 2 <= 1.0

    fat_mask = outer_mask & ~inner_mask
    sos[fat_mask] = 1450.0

    # Fibroglandular tissue inside
    sos[inner_mask] = 1540.0

    # Tumor (circular, offset from center)
    tx, ty = cx + 0.12 * image_size, cy - 0.08 * image_size
    r_tumor = 0.08 * image_size
    tumor_mask = (x - tx) ** 2 + (y - ty) ** 2 <= r_tumor ** 2
    sos[tumor_mask] = 1580.0

    # Cyst (fluid-filled, circular)
    cyst_x, cyst_y = cx - 0.15 * image_size, cy + 0.10 * image_size
    r_cyst = 0.05 * image_size
    cyst_mask = (x - cyst_x) ** 2 + (y - cyst_y) ** 2 <= r_cyst ** 2
    sos[cyst_mask] = 1500.0

    # Calcification (small, high SoS)
    calc_x, calc_y = cx + 0.05 * image_size, cy + 0.18 * image_size
    r_calc = 0.025 * image_size
    calc_mask = (x - calc_x) ** 2 + (y - calc_y) ** 2 <= r_calc ** 2
    sos[calc_mask] = 2500.0

    return sos


def generate_data(task_dir, image_size=128, n_angles=60, noise_std=0.01,
                  seed=42):
    """Generate and save all data for the ultrasound SoS tomography task.

    The forward model uses slowness perturbation delta_s = 1/c - 1/c_water
    rather than absolute slowness, because the Radon transform with circle=True
    only reconstructs within the inscribed circle and sets the exterior to zero.
    Slowness perturbation is naturally zero in water, so this is consistent.

    Parameters
    ----------
    task_dir : str
        Root directory of the task.
    image_size : int
        Phantom size (pixels).
    n_angles : int
        Number of projection angles.
    noise_std : float
        Gaussian noise std relative to sinogram max.
    seed : int
        Random seed.
    """
    data_dir = os.path.join(task_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Generate phantom
    sos_phantom = generate_sos_phantom(image_size)
    background_sos = 1500.0

    # Slowness perturbation: delta_s = 1/c - 1/c_water
    slowness_perturbation = 1.0 / sos_phantom - 1.0 / background_sos

    # Projection angles
    angles = np.linspace(0, 180, n_angles, endpoint=False)

    # Forward model: travel-time sinogram of slowness perturbation
    # t_ij = integral delta_s dl + t_water  (we only measure the perturbation part)
    sinogram_clean = radon_forward(slowness_perturbation, angles)

    # Add noise
    sinogram_noisy = add_gaussian_noise(sinogram_clean, noise_std, rng=rng)

    # Full-angle reference
    n_angles_full = 180
    angles_full = np.linspace(0, 180, n_angles_full, endpoint=False)
    sinogram_full = radon_forward(slowness_perturbation, angles_full)

    # Save ground truth (batch-first)
    np.savez(
        os.path.join(data_dir, "ground_truth.npz"),
        sos_phantom=sos_phantom[np.newaxis, ...],                    # (1, H, W) m/s
        slowness_perturbation=slowness_perturbation[np.newaxis, ...],  # (1, H, W) s/m
    )

    # Save raw data (batch-first)
    np.savez(
        os.path.join(data_dir, "raw_data.npz"),
        sinogram=sinogram_noisy[np.newaxis, ...],        # (1, n_det, n_angles)
        sinogram_clean=sinogram_clean[np.newaxis, ...],  # (1, n_det, n_angles)
        sinogram_full=sinogram_full[np.newaxis, ...],    # (1, n_det, n_angles_full)
        angles=angles[np.newaxis, ...],                  # (1, n_angles)
        angles_full=angles_full[np.newaxis, ...],        # (1, n_angles_full)
    )

    # Save metadata (no solver parameters!)
    meta = {
        "image_size": image_size,
        "n_angles": n_angles,
        "n_angles_full": n_angles_full,
        "noise_std": noise_std,
        "n_detectors": int(sinogram_clean.shape[0]),
        "angle_range_deg": 180.0,
        "pixel_size_mm": 0.5,
        "background_sos_m_per_s": background_sos,
        "ring_radius_mm": 50.0,
        "seed": seed,
    }
    with open(os.path.join(data_dir, "meta_data.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"SoS phantom shape: {sos_phantom.shape}")
    print(f"SoS range: [{sos_phantom.min():.0f}, {sos_phantom.max():.0f}] m/s")
    print(f"Slowness perturbation range: [{slowness_perturbation.min():.6e}, "
          f"{slowness_perturbation.max():.6e}] s/m")
    print(f"Sinogram shape: {sinogram_noisy.shape}")
    print(f"Angles: {n_angles}, Full angles: {n_angles_full}")
    print(f"Data saved to {data_dir}")

    return sos_phantom, slowness_perturbation, sinogram_noisy, sinogram_clean, angles


if __name__ == "__main__":
    task_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_data(task_dir)
