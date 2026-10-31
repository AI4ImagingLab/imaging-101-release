"""
Recorruption scheme: generate self-supervised training pairs from a single noisy image.

The key insight of ZS-DeconvNet (Qiao et al., Nature Communications 2024) is that
a self-supervised denoiser can be trained from a single noisy image by exploiting the
statistical independence of noise across recorrupted realisations.

From a single noisy image y, two independent noisy versions are constructed:
  y_hat = y + D * g       (input to the network, more noisy)
  y_bar = y - D^{-1} * g  (target for the network, less noisy)

where D = alpha * I is a scaled identity matrix (alpha > 1 controls noise strength),
and g ~ N(0, sigma^2 * I) is a random noise map with the same variance structure
as the original noise:
  sigma^2 = beta1 * max(H(y - bg), 0) + beta2

Because y_hat and y_bar share the same underlying signal but have statistically
independent noise contributions (conditioned on g), minimising MSE(f(y_hat), y_bar)
is equivalent to minimising MSE(f(y), x) in expectation, where x is the clean signal.
"""

import numpy as np
from scipy.ndimage import uniform_filter


def recorrupt(y, beta1, beta2, alpha=1.5, bg=100.0, filter_size=5, rng=None):
    """
    Generate a recorrupted training pair from a single noisy image.

    Parameters
    ----------
    y : np.ndarray, shape (H, W), float64
        Single noisy fluorescence image.
    beta1 : float
        Poissonian factor (estimated from y or set to 1.0).
    beta2 : float
        Gaussian read noise variance.
    alpha : float
        Noise strength modulation (controls how much extra noise is added).
        Must be > 1. Theoretically optimal alpha = 1.
    bg : float
        Background offset (ADU).
    filter_size : int
        Low-pass filter size for signal estimation.
    rng : np.random.Generator or None

    Returns
    -------
    y_hat : np.ndarray, shape (H, W), float64
        More noisy version (used as network input during training).
    y_bar : np.ndarray, shape (H, W), float64
        Less noisy version (used as training target).
    """
    if rng is None:
        rng = np.random.default_rng()
    signal = np.maximum(uniform_filter(y - bg, size=filter_size), 0.0)
    sigma2 = beta1 * signal + beta2
    g = rng.standard_normal(y.shape) * np.sqrt(sigma2)
    y_hat = y + alpha * g
    y_bar = y - g / alpha
    return y_hat, y_bar


def prctile_norm(x, pmin=0.0, pmax=100.0):
    """
    Percentile normalization to [0, 1].

    Parameters
    ----------
    x : np.ndarray
        Input array.
    pmin, pmax : float
        Percentile bounds for normalization (default 0 and 100 = min/max).

    Returns
    -------
    x_norm : np.ndarray, same shape, float32
        Normalized array clipped to [0, 1].
    """
    lo = np.percentile(x, pmin)
    hi = np.percentile(x, pmax)
    if hi == lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def extract_patches(images, patch_size=128, n_patches=10000,
                    beta1=1.0, beta2=0.5, alpha=1.5, bg=100.0,
                    filter_size=5, seed=None):
    """
    Extract random patches from images and generate recorrupted pairs.

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W), float64
        Stack of noisy images to sample patches from.
    patch_size : int
        Spatial size of each square patch (default 128).
    n_patches : int
        Total number of (y_hat, y_bar) patch pairs to generate.
    beta1, beta2, alpha, bg, filter_size : float
        Noise model and recorruption parameters.
    seed : int or None

    Returns
    -------
    y_hat_patches : np.ndarray, shape (N, 1, patch_size, patch_size), float32
        More-noisy input patches, normalised to [0, 1].
    y_bar_patches : np.ndarray, shape (N, 1, patch_size, patch_size), float32
        Less-noisy target patches, normalised to [0, 1].
    norm_stats : list of (lo, hi) per patch
        Stored for optional denormalisation.
    """
    rng = np.random.default_rng(seed)
    N_img, H, W = images.shape
    patches_per_img = max(1, n_patches // N_img)
    y_hat_list, y_bar_list = [], []

    for img in images:
        img = img.astype(np.float64)
        for _ in range(patches_per_img):
            # Random crop
            i = rng.integers(0, H - patch_size)
            j = rng.integers(0, W - patch_size)
            patch = img[i:i+patch_size, j:j+patch_size]

            # Augment: random rotation (0, 90, 180, 270) and horizontal flip
            k = rng.integers(4)
            patch = np.rot90(patch, k)
            if rng.random() < 0.5:
                patch = np.fliplr(patch)

            # Recorrupt
            y_hat, y_bar = recorrupt(patch, beta1, beta2, alpha, bg,
                                     filter_size, rng=rng)

            # Normalise each pair jointly using the input statistics
            lo = np.percentile(y_hat, 0)
            hi = np.percentile(y_hat, 100)
            eps = 1e-6
            rng_span = hi - lo if hi > lo else eps
            y_hat_n = np.clip((y_hat - lo) / rng_span, 0.0, 1.0)
            y_bar_n = np.clip((y_bar - lo) / rng_span, 0.0, 1.0)

            y_hat_list.append(y_hat_n[np.newaxis].astype(np.float32))
            y_bar_list.append(y_bar_n[np.newaxis].astype(np.float32))

    y_hat_patches = np.stack(y_hat_list)   # (N, 1, H, W)
    y_bar_patches = np.stack(y_bar_list)   # (N, 1, H, W)
    return y_hat_patches, y_bar_patches
