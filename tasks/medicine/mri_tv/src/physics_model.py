"""
MRI Forward Model
=================

Implements the multi-coil MRI forward operator:

    y_c = M * F * S_c * x,   c = 1, ..., C

where:
    x     : complex image (H, W)
    S_c   : coil sensitivity map for coil c
    F     : 2D FFT (ortho-normalized)
    M     : binary undersampling mask (phase-encode direction)
    y_c   : measured k-space for coil c

The adjoint (zero-filled reconstruction) combines coils via MVUE:

    x_adj = sum_c conj(S_c) * F^H * y_c / sqrt(sum_c |S_c|^2)

Reference
---------
InverseBench (Wu et al.), multi-coil MRI forward operator.
"""

import numpy as np


def fft2c(x: np.ndarray) -> np.ndarray:
    """
    Centered 2D FFT with ortho normalization.

    Parameters
    ----------
    x : ndarray, (..., H, W)
        Image-domain data.

    Returns
    -------
    ndarray, (..., H, W)
        k-space data.
    """
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def ifft2c(x: np.ndarray) -> np.ndarray:
    """
    Centered 2D inverse FFT with ortho normalization.

    Parameters
    ----------
    x : ndarray, (..., H, W)
        k-space data.

    Returns
    -------
    ndarray, (..., H, W)
        Image-domain data.
    """
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def forward_operator(image, sensitivity_maps, mask):
    """
    Multi-coil MRI forward operator: image -> undersampled k-space.

    Parameters
    ----------
    image : ndarray, (H, W) complex
        Complex image to encode.
    sensitivity_maps : ndarray, (C, H, W) complex
        Coil sensitivity maps.
    mask : ndarray, (W,) float or bool
        1-D undersampling mask (phase-encode direction, applied vertically).

    Returns
    -------
    masked_kspace : ndarray, (C, H, W) complex
        Undersampled multi-coil k-space.
    """
    coil_images = sensitivity_maps * image[None, :, :]
    kspace = fft2c(coil_images)
    mask_2d = mask[None, None, :]  # (1, 1, W)
    return kspace * mask_2d


def adjoint_operator(masked_kspace, sensitivity_maps):
    """
    Multi-coil MRI adjoint: undersampled k-space -> MVUE image estimate.

    Parameters
    ----------
    masked_kspace : ndarray, (C, H, W) complex
        Undersampled multi-coil k-space.
    sensitivity_maps : ndarray, (C, H, W) complex
        Coil sensitivity maps.

    Returns
    -------
    image : ndarray, (H, W) complex
        MVUE combined image estimate.
    """
    coil_images = ifft2c(masked_kspace)
    combined = np.sum(coil_images * np.conj(sensitivity_maps), axis=0)
    normalization = np.sqrt(np.sum(np.abs(sensitivity_maps) ** 2, axis=0))
    normalization = np.maximum(normalization, 1e-12)
    return combined / normalization


def generate_undersampling_mask(
    total_lines: int = 320,
    acceleration_ratio: int = 8,
    acs_fraction: float = 0.04,
    pattern: str = "random",
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a 1-D phase-encode undersampling mask.

    Parameters
    ----------
    total_lines : int
        Total number of phase-encode lines.
    acceleration_ratio : int
        Undersampling factor (e.g., 4 or 8).
    acs_fraction : float
        Fraction of center k-space lines to always sample (ACS lines).
        Default 0.04 for acceleration >= 7, 0.08 for lower.
    pattern : str
        'random' or 'equispaced'.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    mask : ndarray, (total_lines,) float32
        Binary mask (1 = sampled, 0 = not sampled).
    """
    rng = np.random.RandomState(seed)
    acs_lines = int(np.floor(acs_fraction * total_lines))
    num_sampled = int(np.floor(total_lines / acceleration_ratio))

    center_start = (total_lines - acs_lines) // 2
    center_idx = np.arange(center_start, center_start + acs_lines)
    outer_idx = np.setdiff1d(np.arange(total_lines), center_idx)

    if pattern == "random":
        random_idx = rng.choice(outer_idx, size=int(num_sampled - acs_lines), replace=False)
    elif pattern == "equispaced":
        random_idx = outer_idx[:: int(acceleration_ratio)]
    else:
        raise ValueError(f"Unknown mask pattern: {pattern}")

    mask = np.zeros(total_lines, dtype=np.float32)
    mask[center_idx] = 1.0
    mask[random_idx] = 1.0
    return mask
