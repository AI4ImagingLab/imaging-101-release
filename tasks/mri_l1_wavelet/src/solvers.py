"""
L1-Wavelet MRI Reconstruction Solver
=====================================

Solves the multi-coil compressed sensing MRI problem:

    argmin_x  (1/2) sum_c ||M F S_c x - y_c||_2^2  +  lambda * ||Psi x||_1

where Psi is a wavelet transform (Daubechies-4), providing multi-resolution
sparsity.

Algorithm: FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) with
Nesterov acceleration. Step size estimated via power iteration on A^H A.

Components extracted from SigPy:
- sigpy/wavelet.py fwt/iwt  → wavelet_forward / wavelet_inverse
  (reimplemented here using convolution + downsampling from filter coefficients)
- sigpy/thresh.py soft_thresh → soft_thresh
- sigpy/mri/linop.py Sense → sense_forward / sense_adjoint / sense_normal
- sigpy/app.py MaxEig → estimate_max_eigenvalue
- sigpy/alg.py GradientMethod → fista_l1_wavelet

Reference
---------
Lustig, M., Donoho, D., & Pauly, J. M. (2007). Sparse MRI.
Beck, A., & Teboulle, M. (2009). FISTA. SIAM J. Imaging Sci.
Mallat, S. (2009). A Wavelet Tour of Signal Processing. Academic Press.
SigPy: https://github.com/mikgroup/sigpy
"""

import numpy as np
from scipy.signal import convolve

from src.physics_model import fft2c, ifft2c


# ---------------------------------------------------------------------------
# Wavelet filter coefficients — loaded from data/wavelet_filters.npz
#
# The filter coefficients (e.g., Daubechies-4) are derived from a
# mathematical construction and stored as data rather than hardcoded.
# This avoids requiring an agent to recall precise floating-point values.
# ---------------------------------------------------------------------------

_FILTER_CACHE = {}


def get_wavelet_filters(wave_name="db4", data_dir="data"):
    """
    Load wavelet filter coefficients from data/wavelet_filters.npz.

    Parameters
    ----------
    wave_name : str
        'db4' or 'haar'.
    data_dir : str
        Directory containing wavelet_filters.npz.

    Returns
    -------
    dec_lo, dec_hi, rec_lo, rec_hi : ndarray (1D)
    """
    if wave_name in _FILTER_CACHE:
        return _FILTER_CACHE[wave_name]

    import os
    path = os.path.join(data_dir, "wavelet_filters.npz")
    filters = np.load(path)

    prefix = wave_name
    keys = [f"{prefix}_dec_lo", f"{prefix}_dec_hi",
            f"{prefix}_rec_lo", f"{prefix}_rec_hi"]
    if keys[0] not in filters:
        raise ValueError(
            f"Unknown wavelet '{wave_name}'. Available prefixes in "
            f"wavelet_filters.npz: check file keys."
        )
    result = tuple(filters[k] for k in keys)
    _FILTER_CACHE[wave_name] = result
    return result


# ---------------------------------------------------------------------------
# 1D wavelet decomposition / reconstruction primitives
# ---------------------------------------------------------------------------

def _dwt1d(signal, dec_lo, dec_hi):
    """
    Single-level 1D discrete wavelet transform (matching pywt 'zero' mode).

    Zero-pads signal, convolves with decomposition filters, downsamples by 2.
    Output length = (N + filter_len - 1) // 2, matching pywt convention.

    Parameters
    ----------
    signal : ndarray, (N,)
    dec_lo, dec_hi : ndarray, (filter_len,)

    Returns
    -------
    approx, detail : ndarray
    """
    filt_len = len(dec_lo)
    n = len(signal)
    out_len = (n + filt_len - 1) // 2

    # Full convolution (output length = N + filt_len - 1)
    lo = convolve(signal, dec_lo, mode="full")
    hi = convolve(signal, dec_hi, mode="full")

    # Downsample by 2 starting at index 1 (matching pywt offset)
    approx = lo[1::2][:out_len]
    detail = hi[1::2][:out_len]

    return approx, detail


def _idwt1d(approx, detail, rec_lo, rec_hi, output_len):
    """
    Single-level 1D inverse DWT (matching pywt 'zero' mode).

    Upsamples by 2 (insert zeros), convolves with reconstruction filters, sums.

    Parameters
    ----------
    approx, detail : ndarray
    rec_lo, rec_hi : ndarray
    output_len : int

    Returns
    -------
    signal : ndarray, (output_len,)
    """
    n = len(approx)
    filt_len = len(rec_lo)

    # Upsample by 2 (zeros in even positions, data in odd)
    up_len = 2 * n
    up_a = np.zeros(up_len, dtype=approx.dtype)
    up_a[1::2] = approx
    up_d = np.zeros(up_len, dtype=detail.dtype)
    up_d[1::2] = detail

    lo = convolve(up_a, rec_lo, mode="full")
    hi = convolve(up_d, rec_hi, mode="full")

    result = lo + hi
    # Trim: skip the first (filt_len - 1) samples, take output_len
    start = filt_len - 1
    return result[start:start + output_len]


# ---------------------------------------------------------------------------
# 2D wavelet decomposition / reconstruction (separable)
# ---------------------------------------------------------------------------

def _dwt2d(image, dec_lo, dec_hi):
    """
    Single-level 2D DWT (separable: rows then columns).

    Parameters
    ----------
    image : ndarray, (H, W) — can be complex
    dec_lo, dec_hi : ndarray, (filter_len,)

    Returns
    -------
    LL, LH, HL, HH : ndarray — four subbands
    """
    H, W = image.shape
    filt_len = len(dec_lo)
    out_w = (W + filt_len - 1) // 2
    out_h = (H + filt_len - 1) // 2

    # Apply DWT along rows (axis=1)
    L_rows = np.zeros((H, out_w), dtype=image.dtype)
    H_rows = np.zeros((H, out_w), dtype=image.dtype)
    for i in range(H):
        L_rows[i], H_rows[i] = _dwt1d(image[i], dec_lo, dec_hi)

    # Apply DWT along columns (axis=0)
    LL = np.zeros((out_h, out_w), dtype=image.dtype)
    LH = np.zeros((out_h, out_w), dtype=image.dtype)
    HL = np.zeros((out_h, out_w), dtype=image.dtype)
    HH = np.zeros((out_h, out_w), dtype=image.dtype)

    for j in range(out_w):
        LL[:, j], LH[:, j] = _dwt1d(L_rows[:, j], dec_lo, dec_hi)
        HL[:, j], HH[:, j] = _dwt1d(H_rows[:, j], dec_lo, dec_hi)

    return LL, LH, HL, HH


def _idwt2d(LL, LH, HL, HH, rec_lo, rec_hi, output_shape):
    """
    Single-level 2D inverse DWT (separable: columns then rows).

    Parameters
    ----------
    LL, LH, HL, HH : ndarray — four subbands
    rec_lo, rec_hi : ndarray
    output_shape : tuple (H, W)

    Returns
    -------
    image : ndarray, (H, W)
    """
    H, W = output_shape
    n_cols = LL.shape[1]  # subband width (same for all 4)

    # Inverse DWT along columns (axis=0): reconstruct to H rows
    L_rows = np.zeros((H, n_cols), dtype=LL.dtype)
    H_rows = np.zeros((H, n_cols), dtype=HL.dtype)

    for j in range(n_cols):
        L_rows[:, j] = _idwt1d(LL[:, j], LH[:, j], rec_lo, rec_hi, H)
        H_rows[:, j] = _idwt1d(HL[:, j], HH[:, j], rec_lo, rec_hi, H)

    # Inverse DWT along rows (axis=1): reconstruct to W columns
    image = np.zeros((H, W), dtype=LL.dtype)
    for i in range(H):
        image[i] = _idwt1d(L_rows[i], H_rows[i], rec_lo, rec_hi, W)

    return image


# ---------------------------------------------------------------------------
# Multi-level 2D wavelet transform with packing/unpacking
# ---------------------------------------------------------------------------

def wavelet_forward(image, wave_name="db4", level=None):
    """
    Multi-level 2D forward wavelet transform.

    Decomposes image into approximation + detail coefficients at multiple
    scales, then packs them into a single array.

    Parameters
    ----------
    image : ndarray, (H, W) — can be complex
    wave_name : str
    level : int or None
        Number of decomposition levels. If None, uses max level.

    Returns
    -------
    coeffs_array : ndarray
        Packed wavelet coefficients.
    coeff_info : list of dict
        Metadata for unpacking (shapes and offsets at each level).
    original_shape : tuple
        Original image shape.
    """
    dec_lo, dec_hi, _, _ = get_wavelet_filters(wave_name)

    H, W = image.shape
    if level is None:
        filt_len = len(dec_lo)
        level = max(1, int(np.floor(np.log2(min(H, W) / (filt_len - 1)))))

    # Decompose
    approx = image
    details = []
    input_shapes = [image.shape]  # shape fed into each level's DWT

    for _ in range(level):
        LL, LH, HL, HH = _dwt2d(approx, dec_lo, dec_hi)
        details.append({"LH": LH, "HL": HL, "HH": HH})
        input_shapes.append(approx.shape)
        approx = LL

    # Pack into single array: [LL | LH(coarsest) ... HH(coarsest) | ... | LH(finest) ... HH(finest)]
    all_coeffs = [approx.ravel()]
    coeff_info = [{"type": "approx", "shape": approx.shape, "n_levels": level,
                   "input_shapes": input_shapes}]

    for i in range(level - 1, -1, -1):
        for key in ["LH", "HL", "HH"]:
            band = details[i][key]
            all_coeffs.append(band.ravel())
            coeff_info.append({"type": key, "shape": band.shape, "level": i})

    coeffs_array = np.concatenate(all_coeffs)
    return coeffs_array, coeff_info, image.shape


def wavelet_inverse(coeffs_array, coeff_info, original_shape,
                    wave_name="db4"):
    """
    Multi-level 2D inverse wavelet transform.

    Parameters
    ----------
    coeffs_array : ndarray
        Packed coefficients from wavelet_forward.
    coeff_info : list of dict
        Metadata from wavelet_forward.
    original_shape : tuple
        Original image shape.
    wave_name : str

    Returns
    -------
    image : ndarray, original_shape
    """
    _, _, rec_lo, rec_hi = get_wavelet_filters(wave_name)

    # Unpack coefficients
    offset = 0
    approx_meta = coeff_info[0]
    approx_size = int(np.prod(approx_meta["shape"]))
    approx = coeffs_array[offset:offset + approx_size].reshape(approx_meta["shape"])
    offset += approx_size
    n_levels = approx_meta["n_levels"]
    input_shapes = approx_meta["input_shapes"]

    # Collect detail bands per level
    detail_bands = {}
    for info in coeff_info[1:]:
        lvl = info["level"]
        size = int(np.prod(info["shape"]))
        band = coeffs_array[offset:offset + size].reshape(info["shape"])
        offset += size
        if lvl not in detail_bands:
            detail_bands[lvl] = {}
        detail_bands[lvl][info["type"]] = band

    # Reconstruct from coarsest to finest
    current = approx
    for lvl in range(n_levels - 1, -1, -1):
        LH = detail_bands[lvl]["LH"]
        HL = detail_bands[lvl]["HL"]
        HH = detail_bands[lvl]["HH"]

        # Target shape is the input that was fed into this level's DWT
        target_shape = input_shapes[lvl + 1]

        current = _idwt2d(current, LH, HL, HH, rec_lo, rec_hi, target_shape)

    return current[:original_shape[0], :original_shape[1]]


# ---------------------------------------------------------------------------
# Soft thresholding (extracted from sigpy/thresh.py)
# ---------------------------------------------------------------------------

def soft_thresh(lamda, x):
    """
    Complex soft thresholding: (|x| - lambda)_+ * sign(x).
    """
    abs_x = np.abs(x)
    sign = np.where(abs_x > 0, x / abs_x, 0)
    mag = np.maximum(abs_x - lamda, 0)
    return mag * sign


# ---------------------------------------------------------------------------
# SENSE normal operator A^H A
# ---------------------------------------------------------------------------

def sense_forward(image, sensitivity_maps, mask):
    """SENSE forward: image → undersampled multi-coil k-space."""
    coil_images = sensitivity_maps * image[None, :, :]
    kspace = fft2c(coil_images)
    return kspace * mask[None, None, :]


def sense_adjoint(kspace, sensitivity_maps):
    """SENSE adjoint: multi-coil k-space → image."""
    coil_images = ifft2c(kspace)
    return np.sum(np.conj(sensitivity_maps) * coil_images, axis=0)


def sense_normal(image, sensitivity_maps, mask):
    """SENSE normal operator: A^H A (x)."""
    return sense_adjoint(sense_forward(image, sensitivity_maps, mask),
                         sensitivity_maps)


# ---------------------------------------------------------------------------
# Power iteration for step size
# ---------------------------------------------------------------------------

def estimate_max_eigenvalue(sensitivity_maps, mask, max_iter=30):
    """Estimate max eigenvalue of A^H A via power iteration."""
    H, W = sensitivity_maps.shape[1:]
    x = np.random.randn(H, W).astype(np.complex128)
    x /= np.linalg.norm(x)
    for _ in range(max_iter):
        y = sense_normal(x, sensitivity_maps, mask)
        max_eig = np.linalg.norm(y)
        if max_eig > 0:
            x = y / max_eig
        else:
            break
    return float(max_eig)


# ---------------------------------------------------------------------------
# FISTA solver
# ---------------------------------------------------------------------------

def fista_l1_wavelet(
    masked_kspace, sensitivity_maps, mask,
    lamda=1e-3, wave_name="db4", max_iter=100, accelerate=True,
):
    """
    FISTA for L1-wavelet CS-MRI.

    Solves: min_x (1/2)||Ax - y||^2 + lambda ||W x||_1
    """
    H, W = sensitivity_maps.shape[1:]
    AHy = sense_adjoint(masked_kspace, sensitivity_maps)
    alpha = 1.0 / estimate_max_eigenvalue(sensitivity_maps, mask)

    # Get wavelet packing info (compute once)
    dummy_coeffs, coeff_info, orig_shape = wavelet_forward(
        np.zeros((H, W), dtype=np.complex128), wave_name,
    )

    x = np.zeros((H, W), dtype=masked_kspace.dtype)
    if accelerate:
        z = x.copy()
        t = 1.0

    for _ in range(max_iter):
        x_old = x.copy()
        x_input = z if accelerate else x

        gradf = sense_normal(x_input, sensitivity_maps, mask) - AHy
        x = x_input - alpha * gradf

        coeffs, _, _ = wavelet_forward(x, wave_name)
        coeffs = soft_thresh(lamda * alpha, coeffs)
        x = wavelet_inverse(coeffs, coeff_info, (H, W), wave_name)

        if accelerate:
            t_old = t
            t = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
            z = x + ((t_old - 1) / t) * (x - x_old)

    return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def l1_wavelet_reconstruct_single(
    masked_kspace, sensitivity_maps, lamda=1e-3, wave_name="db4",
    max_iter=100,
):
    """Reconstruct a single MRI image using L1-Wavelet regularization."""
    mask = (np.abs(masked_kspace[0]).sum(axis=0) > 0).astype(np.float32)
    return fista_l1_wavelet(
        masked_kspace, sensitivity_maps, mask,
        lamda=lamda, wave_name=wave_name, max_iter=max_iter,
    )


def l1_wavelet_reconstruct_batch(
    masked_kspace, sensitivity_maps, lamda=1e-3, wave_name="db4",
    max_iter=100,
):
    """Reconstruct a batch of MRI images using L1-Wavelet regularization."""
    n_samples = masked_kspace.shape[0]
    recons = []
    for i in range(n_samples):
        recon = l1_wavelet_reconstruct_single(
            masked_kspace[i], sensitivity_maps[i], lamda, wave_name, max_iter,
        )
        recons.append(recon)
    return np.stack(recons, axis=0)
