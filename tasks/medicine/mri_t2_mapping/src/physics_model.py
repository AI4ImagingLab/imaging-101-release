"""
Multi-echo spin-echo MRI signal model.

Forward model:
    S(TE) = M0 * exp(-TE / T2)

With Rician noise (magnitude of complex Gaussian):
    S_noisy = |S_clean + eta|,  eta ~ CN(0, sigma^2)
"""

import numpy as np


def mono_exponential_signal(M0, T2, TE):
    """Compute mono-exponential T2 decay signal.

    Parameters
    ----------
    M0 : np.ndarray
        Proton density map, shape (...,).
    T2 : np.ndarray
        T2 relaxation time map in ms, shape (...,).
    TE : np.ndarray
        Echo times in ms, shape (N_echoes,).

    Returns
    -------
    signal : np.ndarray
        Multi-echo signal, shape (..., N_echoes).
    """
    M0 = np.asarray(M0, dtype=np.float64)
    T2 = np.asarray(T2, dtype=np.float64)
    TE = np.asarray(TE, dtype=np.float64)

    # Expand TE to broadcast: (..., 1) and (N_echoes,)
    # Safe division: where T2 == 0, signal is 0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = TE[np.newaxis, ...] / T2[..., np.newaxis]
    ratio = np.where(np.isfinite(ratio), ratio, np.inf)

    signal = M0[..., np.newaxis] * np.exp(-ratio)
    signal = np.where(T2[..., np.newaxis] > 0, signal, 0.0)
    return signal


def add_rician_noise(signal, sigma, rng=None):
    """Add Rician noise to magnitude MRI signal.

    Rician noise arises from taking the magnitude of complex data
    with independent Gaussian noise on real and imaginary channels.

    Parameters
    ----------
    signal : np.ndarray
        Clean signal, any shape.
    sigma : float
        Noise standard deviation per channel (real and imaginary).
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    noisy : np.ndarray
        Noisy magnitude signal, same shape as input.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise_real = rng.normal(0, sigma, signal.shape)
    noise_imag = rng.normal(0, sigma, signal.shape)
    noisy = np.sqrt((signal + noise_real) ** 2 + noise_imag ** 2)
    return noisy


def simulate_multi_echo(M0, T2, TE, sigma=0.0, rng=None):
    """Simulate multi-echo spin-echo MRI acquisition.

    Parameters
    ----------
    M0 : np.ndarray
        Proton density map.
    T2 : np.ndarray
        T2 map in ms.
    TE : np.ndarray
        Echo times in ms.
    sigma : float
        Rician noise level. If 0, no noise is added.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    signal : np.ndarray
        Multi-echo signal with shape (*M0.shape, N_echoes).
    """
    signal = mono_exponential_signal(M0, T2, TE)
    if sigma > 0:
        signal = add_rician_noise(signal, sigma, rng=rng)
    return signal
