"""
CT forward model and Poisson noise simulation.

Implements the parallel-beam Radon transform via SVMBIR's projection
operator, and the Poisson noise model for low-dose CT.
"""

import numpy as np
import svmbir


def radon_forward(image: np.ndarray,
                  angles: np.ndarray,
                  num_channels: int) -> np.ndarray:
    """Compute parallel-beam forward projection (Radon transform).

    Uses SVMBIR's built-in projector for consistency with the reconstruction.

    Args:
        image: 2D image of shape (H, W).
        angles: 1D array of projection angles in radians.
        num_channels: Number of detector channels.

    Returns:
        Sinogram of shape (num_views, num_channels).
    """
    image_3d = image[np.newaxis, ...]  # (1, H, W) for SVMBIR
    sino_3d = svmbir.project(image_3d, angles, num_channels, verbose=0)
    # sino_3d shape: (num_views, 1, num_channels) -> squeeze slice dim
    return sino_3d[:, 0, :]


def radon_backproject(sinogram: np.ndarray,
                      angles: np.ndarray,
                      num_rows: int,
                      num_cols: int) -> np.ndarray:
    """Compute parallel-beam back-projection.

    Args:
        sinogram: 2D sinogram of shape (num_views, num_channels).
        angles: 1D array of projection angles in radians.
        num_rows: Number of rows in output image.
        num_cols: Number of columns in output image.

    Returns:
        Back-projected image of shape (num_rows, num_cols).
    """
    sino_3d = sinogram[:, np.newaxis, :]  # (V, 1, C)
    bp_3d = svmbir.backproject(sino_3d, angles, num_rows=num_rows,
                               num_cols=num_cols, verbose=0)
    return bp_3d[0]  # (H, W)


def poisson_pre_log_model(sinogram_clean: np.ndarray,
                          I0: float) -> np.ndarray:
    """Compute expected photon counts from clean sinogram (noiseless).

    I_i = I0 * exp(-[A x]_i)

    Args:
        sinogram_clean: Clean line integrals, any shape.
        I0: Incident photon count per ray.

    Returns:
        Expected photon counts, same shape as sinogram_clean.
    """
    return I0 * np.exp(-sinogram_clean)


def simulate_poisson_noise(transmission: np.ndarray,
                           rng: np.random.RandomState) -> np.ndarray:
    """Draw Poisson-distributed photon counts.

    Args:
        transmission: Expected counts (lambda parameter), any shape.
        rng: NumPy RandomState for reproducibility.

    Returns:
        Noisy photon counts, same shape. Minimum value is 1 (to avoid log(0)).
    """
    counts = rng.poisson(transmission).astype(np.float64)
    return np.maximum(counts, 1.0)


def post_log_transform(photon_counts: np.ndarray, I0: float) -> np.ndarray:
    """Convert photon counts to post-log sinogram.

    y_i = -log(I_i / I0)

    Args:
        photon_counts: Measured photon counts (>= 1).
        I0: Incident photon count.

    Returns:
        Post-log sinogram values.
    """
    return -np.log(photon_counts / I0)


def compute_poisson_weights(photon_counts: np.ndarray) -> np.ndarray:
    """Compute PWLS weights from photon counts.

    In the post-log domain, the variance of measurement y_i is approximately
    1/I_i, so the optimal weight is w_i = I_i (the photon count).

    Args:
        photon_counts: Measured photon counts (>= 1).

    Returns:
        Weights array, same shape as photon_counts.
    """
    return photon_counts.copy()
