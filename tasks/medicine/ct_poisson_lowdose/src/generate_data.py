"""
Synthetic low-dose CT data generation with Poisson noise.

Generates a 2D Shepp-Logan phantom, computes clean sinograms via SVMBIR
forward projection, and simulates Poisson-noise measurements at two dose
levels (low and high).
"""

import json
import os

import numpy as np
import svmbir


# ---------------------------------------------------------------------------
# Physical / acquisition constants
# ---------------------------------------------------------------------------
_IMAGE_SIZE = 256
_NUM_VIEWS = 256
_NUM_CHANNELS = 367
_ANGLE_RANGE = np.pi  # parallel beam, 0 to pi
_ATTENUATION_SCALE = 0.02  # scale Shepp-Logan to realistic mu values (cm^-1)
_I0_LOW = 1000  # low-dose incident photon count
_I0_HIGH = 50000  # high-dose incident photon count
_RANDOM_SEED = 42


def generate_phantom(image_size: int = _IMAGE_SIZE,
                     scale: float = _ATTENUATION_SCALE) -> np.ndarray:
    """Generate a scaled 2D Shepp-Logan phantom.

    Args:
        image_size: Number of rows/cols (square image).
        scale: Multiplicative scale to convert Shepp-Logan values to
            realistic linear attenuation coefficients (cm^-1).

    Returns:
        2D array of shape (image_size, image_size).
    """
    phantom = svmbir.phantom.gen_shepp_logan(image_size, image_size)
    return phantom * scale


def generate_angles(num_views: int = _NUM_VIEWS,
                    angle_range: float = _ANGLE_RANGE) -> np.ndarray:
    """Generate uniformly-spaced projection angles.

    Args:
        num_views: Number of angular views.
        angle_range: Total angular range in radians.

    Returns:
        1D array of angles in radians, shape (num_views,).
    """
    return np.linspace(0, angle_range, num_views, endpoint=False)


def forward_project(phantom_3d: np.ndarray,
                    angles: np.ndarray,
                    num_channels: int = _NUM_CHANNELS) -> np.ndarray:
    """Compute clean sinogram via SVMBIR parallel-beam forward projection.

    Args:
        phantom_3d: Phantom with shape (1, H, W).
        angles: 1D angle array in radians.
        num_channels: Number of detector channels.

    Returns:
        Sinogram array of shape (num_views, 1, num_channels).
    """
    return svmbir.project(phantom_3d, angles, num_channels, verbose=0)


def simulate_poisson_sinogram(sino_clean: np.ndarray,
                              I0: float,
                              rng: np.random.RandomState
                              ) -> tuple:
    """Simulate Poisson-noise sinogram from clean line integrals.

    Pre-log model:  I_i ~ Poisson(I0 * exp(-[A x]_i))
    Post-log data:  y_i = -log(I_i / I0)
    Weights:        w_i = I_i  (inverse of post-log variance)

    Args:
        sino_clean: Clean sinogram (line integrals), shape (V, S, C).
        I0: Incident photon count (scalar).
        rng: NumPy RandomState for reproducibility.

    Returns:
        (sino_noisy, weights, photon_counts) where
        - sino_noisy: post-log noisy sinogram, same shape
        - weights: Poisson-derived weights (= photon counts), same shape
        - photon_counts: raw noisy photon counts before log
    """
    transmission = I0 * np.exp(-sino_clean)
    photon_counts = rng.poisson(transmission).astype(np.float64)
    # Clamp to >= 1 to avoid log(0)
    photon_counts = np.maximum(photon_counts, 1.0)
    sino_noisy = -np.log(photon_counts / I0)
    weights = photon_counts.copy()
    return sino_noisy, weights, photon_counts


def generate_all(output_dir: str) -> None:
    """Generate all data files for the ct_poisson_lowdose task.

    Creates:
        data/raw_data.npz
        data/ground_truth.npz
        data/meta_data.json
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Phantom
    phantom = generate_phantom()
    phantom_3d = phantom[np.newaxis, ...]  # (1, 256, 256)

    # 2. Angles
    angles = generate_angles()

    # 3. Clean sinogram
    sino_clean = forward_project(phantom_3d, angles)
    # sino_clean shape: (num_views, 1, num_channels)

    # 4. Poisson noisy sinograms
    rng = np.random.RandomState(_RANDOM_SEED)
    sino_noisy_low, weights_low, _ = simulate_poisson_sinogram(
        sino_clean, _I0_LOW, rng)
    # Use a different seed for high dose to get independent noise
    rng_high = np.random.RandomState(_RANDOM_SEED + 1)
    sino_noisy_high, weights_high, _ = simulate_poisson_sinogram(
        sino_clean, _I0_HIGH, rng_high)

    # 5. Save ground truth (batch-first)
    np.savez(
        os.path.join(output_dir, "ground_truth.npz"),
        phantom=phantom_3d.astype(np.float64),        # (1, 256, 256)
    )

    # 6. Save raw data (batch-first)
    # Sinogram convention: transpose to (1, num_views, num_channels)
    # SVMBIR uses (views, slices, channels); squeeze slice dim and add batch
    def _to_batch_first(sino):
        """(V, 1, C) -> (1, V, C)"""
        return sino[:, 0, :][np.newaxis, ...]  # (1, V, C)

    np.savez(
        os.path.join(output_dir, "raw_data.npz"),
        sinogram_clean=_to_batch_first(sino_clean).astype(np.float64),
        sinogram_low_dose=_to_batch_first(sino_noisy_low).astype(np.float64),
        sinogram_high_dose=_to_batch_first(sino_noisy_high).astype(np.float64),
        weights_low_dose=_to_batch_first(weights_low).astype(np.float64),
        weights_high_dose=_to_batch_first(weights_high).astype(np.float64),
        angles=angles[np.newaxis, ...].astype(np.float64),  # (1, num_views)
    )

    # 7. Save metadata (no solver params)
    meta = {
        "image_size": _IMAGE_SIZE,
        "num_views": _NUM_VIEWS,
        "num_channels": _NUM_CHANNELS,
        "angle_range_rad": float(_ANGLE_RANGE),
        "geometry": "parallel",
        "delta_channel": 1.0,
        "delta_pixel": 1.0,
        "attenuation_scale": float(_ATTENUATION_SCALE),
        "I0_low_dose": int(_I0_LOW),
        "I0_high_dose": int(_I0_HIGH),
        "random_seed": _RANDOM_SEED,
    }
    with open(os.path.join(output_dir, "meta_data.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Data saved to {output_dir}")
    print(f"  phantom: {phantom_3d.shape}")
    print(f"  sinogram_clean: {_to_batch_first(sino_clean).shape}")
    print(f"  sinogram_low_dose: {_to_batch_first(sino_noisy_low).shape}")
    print(f"  sinogram_high_dose: {_to_batch_first(sino_noisy_high).shape}")


if __name__ == "__main__":
    task_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_all(os.path.join(task_dir, "data"))
