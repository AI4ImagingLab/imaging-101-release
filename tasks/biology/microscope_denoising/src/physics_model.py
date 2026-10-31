"""
Forward model: Poisson-Gaussian noise for fluorescence microscopy.

Fluorescence microscopy images are corrupted by mixed Poisson-Gaussian noise:
  - Shot noise (Poissonian): variance proportional to signal intensity
  - Read noise (Gaussian): fixed variance from detector electronics

In the Gaussian approximation valid for sCMOS cameras, the joint model is:
  y = x + n,  n ~ N(0, sigma^2)
  sigma^2(x) = beta1 * max(H(x - bg), 0) + beta2

where H is a 5x5 averaging filter that estimates local signal intensity,
beta1 is the Poissonian scaling factor, beta2 is the read noise variance,
and bg is the camera background offset.
"""

import numpy as np
from scipy.ndimage import uniform_filter, convolve


def noise_variance(x, beta1, beta2, bg=100.0, filter_size=5):
    """
    Compute pixel-wise noise variance under the Poisson-Gaussian model.

    Parameters
    ----------
    x : np.ndarray, shape (..., H, W), float64
        Fluorescence image (ADU, including background offset).
    beta1 : float
        Poissonian factor (signal-dependent noise). Theoretically optimal = 1.
    beta2 : float
        Gaussian factor (read noise variance). Camera-dependent, typically 0.5-3.
    bg : float
        Background offset in ADU (camera bias + dark current).
    filter_size : int
        Size of the averaging filter used to estimate local signal.

    Returns
    -------
    sigma2 : np.ndarray, shape (..., H, W), float64
        Per-pixel noise variance.
    """
    signal = uniform_filter(x - bg, size=filter_size)
    signal = np.maximum(signal, 0.0)
    return beta1 * signal + beta2


def add_noise(x, beta1, beta2, bg=100.0, filter_size=5, rng=None):
    """
    Generate a noisy observation from a fluorescence image.

    Parameters
    ----------
    x : np.ndarray, shape (..., H, W), float64
        Clean fluorescence signal (ADU).
    beta1 : float
        Poissonian factor.
    beta2 : float
        Gaussian read noise variance.
    bg : float
        Background offset (ADU).
    filter_size : int
        Low-pass filter size for signal estimation.
    rng : np.random.Generator or None
        Random number generator. If None, uses default.

    Returns
    -------
    y : np.ndarray, shape (..., H, W), float64
        Noisy observation.
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma2 = noise_variance(x, beta1, beta2, bg, filter_size)
    noise = rng.standard_normal(x.shape) * np.sqrt(sigma2)
    return x + noise


def load_psf(psf_path, kernel_size=32):
    """
    Load a PSF from a TIFF file, center-crop, and normalise to unit sum.

    Parameters
    ----------
    psf_path : str
        Path to a 2D PSF image stored as TIFF (e.g., from Zenodo dataset).
    kernel_size : int
        Output kernel size (square).  The PSF is cropped around its centroid.

    Returns
    -------
    psf : np.ndarray, shape (kernel_size, kernel_size), float32
        Normalised PSF kernel.
    """
    import tifffile
    psf_raw = tifffile.imread(psf_path).astype(np.float64)
    if psf_raw.ndim > 2:
        psf_raw = psf_raw[psf_raw.shape[0] // 2]  # take central slice
    cy, cx = psf_raw.shape[0] // 2, psf_raw.shape[1] // 2
    half = kernel_size // 2
    psf = psf_raw[cy - half:cy + half, cx - half:cx + half]
    psf = np.maximum(psf, 0.0)
    psf /= psf.sum()
    return psf.astype(np.float32)


def psf_convolve(x, psf):
    """
    Convolve image x with a PSF kernel (spatial domain, reflect padding).

    Parameters
    ----------
    x : np.ndarray, shape (H, W), float64
        Input image.
    psf : np.ndarray, shape (kH, kW), float32
        Normalised PSF kernel (sum = 1).

    Returns
    -------
    blurred : np.ndarray, shape (H, W), float64
    """
    return convolve(x.astype(np.float64), psf.astype(np.float64), mode='reflect')


def estimate_noise_params(y, bg=100.0, filter_size=5):
    """
    Estimate beta1 and beta2 from a single noisy image.

    Uses the relationship: Var(y) = beta1 * E[max(H(y-bg), 0)] + beta2.
    Splits the image into blocks, estimates local variance and local mean,
    then fits a linear model: local_var = beta1 * local_mean + beta2.

    Parameters
    ----------
    y : np.ndarray, shape (H, W), float64
        Single noisy image.
    bg : float
        Background offset (ADU).
    filter_size : int
        Low-pass filter size.

    Returns
    -------
    beta1 : float
    beta2 : float
    """
    from scipy.stats import linregress
    H, W = y.shape[-2], y.shape[-1]
    block_size = 32
    means, variances = [], []
    for i in range(0, H - block_size, block_size):
        for j in range(0, W - block_size, block_size):
            block = y[..., i:i+block_size, j:j+block_size].astype(np.float64)
            smooth = uniform_filter(block, size=filter_size)
            signal_est = np.maximum(smooth - bg, 0.0).mean()
            residual = block - smooth
            var_est = residual.var()
            means.append(signal_est)
            variances.append(var_est)
    means = np.array(means)
    variances = np.array(variances)
    # Linear fit: var = beta1 * mean + beta2
    slope, intercept, _, _, _ = linregress(means, variances)
    beta1 = max(slope, 0.1)
    beta2 = max(intercept, 0.1)
    return beta1, beta2
