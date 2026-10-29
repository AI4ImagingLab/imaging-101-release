import numpy as np


def load_and_preprocess_data(raw_signal, nu_axis, noise_level=0.0):
    """
    Loads raw spectral data and performs normalization/preprocessing.

    Args:
        raw_signal (np.ndarray): The measured intensity array.
        nu_axis (np.ndarray): The wavenumber axis.
        noise_level (float): Std dev of Gaussian noise to add (for simulation).

    Returns:
        tuple: (processed_signal, nu_axis)
    """
    signal = np.array(raw_signal, dtype=float)

    # Background subtraction
    bg = 0.0
    signal = signal - bg
    signal[signal < 0] = 0

    # Add noise if requested
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, signal.shape)
        signal = signal + noise
        signal[signal < 0] = 0

    # Normalize
    mx = signal.max()
    if mx > 0:
        signal /= mx

    return signal, nu_axis
