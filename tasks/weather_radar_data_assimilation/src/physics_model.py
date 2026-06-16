"""Forward measurement model for sparse weather radar observations.

The observation model is: y = M * x + eta, where M is a binary mask
and eta ~ N(0, sigma^2 I).
"""

import numpy as np
import torch


def make_observation_operator(mask):
    """Return a function that applies the binary mask to an input array.

    Parameters
    ----------
    mask : array-like
        Binary observation mask. Works with both numpy arrays and torch tensors.

    Returns
    -------
    callable
        Function that element-wise multiplies input by mask.
    """
    def operator(x):
        return x * mask
    return operator


def make_noiser(sigma: float):
    """Return a function that adds Gaussian noise with given sigma.

    Parameters
    ----------
    sigma : float
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    callable
        Function that adds N(0, sigma^2) noise to input.
    """
    def noiser(x):
        if isinstance(x, torch.Tensor):
            return x + torch.randn_like(x) * sigma
        return x + np.random.randn(*x.shape).astype(x.dtype) * sigma
    return noiser


def forward_model(x: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Apply full forward model: y = mask * x + N(0, sigma^2).

    Parameters
    ----------
    x : np.ndarray
        Full-resolution input frames.
    mask : np.ndarray
        Binary observation mask.
    sigma : float
        Noise standard deviation.

    Returns
    -------
    np.ndarray
        Sparse noisy observations.
    """
    y = mask * x + np.random.randn(*x.shape).astype(x.dtype) * sigma
    return y
