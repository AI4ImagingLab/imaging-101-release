"""
Synthetic Data Generation for Non-Cartesian MRI
================================================

Generates:
    1. Shepp-Logan phantom (128x128)
    2. Birdcage coil sensitivity maps (4 coils)
    3. Golden-angle radial k-space trajectory
    4. Multi-coil NUFFT forward projection + noise

All data saved in batch-first (1, ...) convention.
"""

import numpy as np
import sigpy as sp
import sigpy.mri


def shepp_logan_phantom(n: int = 128) -> np.ndarray:
    """
    Generate a Shepp-Logan phantom of size (n, n).

    Parameters
    ----------
    n : int
        Image size.

    Returns
    -------
    phantom : ndarray, (n, n) complex128
        Complex-valued phantom (real positive values).
    """
    return sp.shepp_logan((n, n)).astype(np.complex128)


def generate_coil_maps(n_coils: int, image_shape: tuple) -> np.ndarray:
    """
    Generate birdcage coil sensitivity maps.

    Parameters
    ----------
    n_coils : int
        Number of receive coils.
    image_shape : tuple
        (H, W) image dimensions.

    Returns
    -------
    maps : ndarray, (n_coils, H, W) complex128
        Coil sensitivity maps.
    """
    return sigpy.mri.birdcage_maps((n_coils, *image_shape)).astype(np.complex128)


def golden_angle_radial_trajectory(
    n_spokes: int, n_readout: int, image_shape: tuple
) -> np.ndarray:
    """
    Generate a golden-angle radial k-space trajectory.

    Angles follow the golden angle sequence: theta_i = i * pi * (sqrt(5) - 1) / 2.
    Each spoke has n_readout uniformly spaced points from -0.5 to 0.5
    (normalized k-space), scaled to the image dimensions.

    Parameters
    ----------
    n_spokes : int
        Number of radial spokes.
    n_readout : int
        Number of readout points per spoke.
    image_shape : tuple
        (H, W) image dimensions, used to scale the trajectory.

    Returns
    -------
    coord : ndarray, (n_spokes * n_readout, 2) float64
        Non-Cartesian k-space coordinates in [-N/2, N/2].
    """
    golden_angle = np.pi * (np.sqrt(5) - 1) / 2
    angles = np.arange(n_spokes) * golden_angle

    # Readout points from -0.5 to 0.5 (normalized frequency)
    readout = np.linspace(-0.5, 0.5, n_readout, endpoint=False)

    coord = np.zeros((n_spokes, n_readout, 2), dtype=np.float64)
    for i, angle in enumerate(angles):
        coord[i, :, 0] = readout * image_shape[0] * np.cos(angle)
        coord[i, :, 1] = readout * image_shape[1] * np.sin(angle)

    return coord.reshape(-1, 2)


def nufft_forward(image: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """
    Non-Uniform FFT forward operator.

    Parameters
    ----------
    image : ndarray, (H, W) complex
        Image to transform.
    coord : ndarray, (M, 2) float
        Non-Cartesian k-space coordinates.

    Returns
    -------
    kdata : ndarray, (M,) complex
        k-space samples at the given coordinates.
    """
    return sp.nufft(image, coord)


def generate_synthetic_data(
    image_size: int = 128,
    n_coils: int = 4,
    n_spokes: int = 64,
    n_readout: int = 128,
    noise_std: float = 0.005,
    seed: int = 42,
) -> dict:
    """
    Generate complete synthetic non-Cartesian MRI dataset.

    Parameters
    ----------
    image_size : int
        Image dimension (square).
    n_coils : int
        Number of receive coils.
    n_spokes : int
        Number of radial spokes.
    n_readout : int
        Number of readout points per spoke.
    noise_std : float
        Standard deviation of complex Gaussian noise.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        phantom : (image_size, image_size) complex128
        coil_maps : (n_coils, image_size, image_size) complex128
        coord : (n_spokes * n_readout, 2) float64
        kdata : (n_coils, n_spokes * n_readout) complex128
        kdata_clean : (n_coils, n_spokes * n_readout) complex128
    """
    rng = np.random.RandomState(seed)
    image_shape = (image_size, image_size)

    # Generate phantom and coils
    phantom = shepp_logan_phantom(image_size)
    coil_maps = generate_coil_maps(n_coils, image_shape)

    # Generate trajectory
    coord = golden_angle_radial_trajectory(n_spokes, n_readout, image_shape)

    # Forward: NUFFT of each coil image
    kdata_clean = np.zeros((n_coils, coord.shape[0]), dtype=np.complex128)
    for c in range(n_coils):
        coil_image = coil_maps[c] * phantom
        kdata_clean[c] = nufft_forward(coil_image, coord)

    # Add noise
    noise = noise_std * (
        rng.randn(*kdata_clean.shape) + 1j * rng.randn(*kdata_clean.shape)
    ) / np.sqrt(2)
    kdata = kdata_clean + noise

    return {
        "phantom": phantom,
        "coil_maps": coil_maps,
        "coord": coord,
        "kdata": kdata,
        "kdata_clean": kdata_clean,
    }


def save_data(data_dir: str = "data", **kwargs):
    """
    Save generated data to npz files with batch-first convention.

    Parameters
    ----------
    data_dir : str
        Output directory.
    **kwargs : dict
        Override default generation parameters.
    """
    import os
    import json

    os.makedirs(data_dir, exist_ok=True)

    result = generate_synthetic_data(**kwargs)

    # raw_data.npz: observations + instrument parameters
    # All arrays in batch-first convention (1, ...)
    np.savez_compressed(
        os.path.join(data_dir, "raw_data.npz"),
        kdata=result["kdata"][np.newaxis].astype(np.complex64),        # (1, C, M)
        coord=result["coord"][np.newaxis].astype(np.float32),          # (1, M, 2)
        coil_maps=result["coil_maps"][np.newaxis].astype(np.complex64),  # (1, C, H, W)
    )

    # ground_truth.npz: true phantom
    np.savez_compressed(
        os.path.join(data_dir, "ground_truth.npz"),
        phantom=result["phantom"][np.newaxis].astype(np.complex64),    # (1, H, W)
    )

    # meta_data.json: imaging parameters only
    meta = {
        "image_size": [kwargs.get("image_size", 128)] * 2,
        "n_coils": kwargs.get("n_coils", 4),
        "n_spokes": kwargs.get("n_spokes", 64),
        "n_readout": kwargs.get("n_readout", 128),
        "trajectory_type": "golden_angle_radial",
        "noise_std": kwargs.get("noise_std", 0.005),
        "data_source": "synthetic_shepp_logan",
    }
    with open(os.path.join(data_dir, "meta_data.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved data to {data_dir}/")
    print(f"  raw_data.npz: kdata {result['kdata'].shape}, "
          f"coord {result['coord'].shape}, coil_maps {result['coil_maps'].shape}")
    print(f"  ground_truth.npz: phantom {result['phantom'].shape}")


if __name__ == "__main__":
    save_data()
