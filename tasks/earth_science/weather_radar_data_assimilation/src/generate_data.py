"""Synthetic observation generation for weather radar data assimilation."""

import numpy as np


def generate_observations(
    ground_truth: np.ndarray,
    mask_ratio: float = 0.1,
    noise_sigma: float = 0.001,
    seed: int = 42,
) -> tuple:
    """Generate synthetic sparse observations from ground truth.

    Parameters
    ----------
    ground_truth : np.ndarray
        Full-resolution frames, shape (n_frames, H, W), values in [0, 1].
    mask_ratio : float
        Fraction of pixels to observe (default 0.1 = 10%).
    noise_sigma : float
        Standard deviation of additive Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    observations : np.ndarray
        Sparse noisy observations, same shape as ground_truth.
    mask : np.ndarray
        Binary mask, shape (1, H, W) — same spatial mask for all frames.
    """
    rng = np.random.RandomState(seed)
    H, W = ground_truth.shape[-2], ground_truth.shape[-1]
    mask = (rng.rand(1, H, W) < mask_ratio).astype(np.float32)
    noise = rng.randn(*ground_truth.shape).astype(np.float32) * noise_sigma
    observations = np.clip(ground_truth * mask + noise, 0, 1).astype(np.float32)
    return observations, mask
