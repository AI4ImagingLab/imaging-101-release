"""
L1-Wavelet Non-Cartesian MRI Reconstruction Solver
====================================================

Solves the multi-coil compressed sensing MRI problem with non-Cartesian
(radial) sampling:

    argmin_x  (1/2) ||F_NU(S x) - y||_2^2  +  lambda * ||Psi x||_1

where F_NU is the NUFFT, S is sensitivity encoding, and Psi is the
wavelet transform.

Algorithm: FISTA (Fast Iterative Shrinkage-Thresholding) with Nesterov
acceleration. Step size estimated via power iteration on A^H A.

Components extracted from SigPy:
- SENSE-NUFFT normal operator: A^H A where A = F_NU * S
  (sigpy/mri/linop.py Sense class)
- L1-wavelet proximal: soft-threshold in wavelet domain
  (sigpy/prox.py L1Reg + UnitaryTransform, sigpy/wavelet.py fwt/iwt)
- FISTA loop with Nesterov momentum
  (sigpy/alg.py GradientMethod)
- Power iteration for step size
  (sigpy/app.py MaxEig)

The NUFFT itself (sigpy.nufft / sigpy.nufft_adjoint) is used as a
low-level numerical primitive (Kaiser-Bessel gridding), analogous to
using numpy.fft for Cartesian tasks.

Reference
---------
Lustig, Donoho, & Pauly (2007). Sparse MRI.
Beck & Teboulle (2009). A fast iterative shrinkage-thresholding algorithm.
SigPy: https://github.com/mikgroup/sigpy
"""

import numpy as np
from scipy.signal import convolve

from src.physics_model import (
    nufft_forward, nufft_adjoint,
    multicoil_nufft_forward,
)


# ---------------------------------------------------------------------------
# Wavelet filter coefficients — loaded from data/wavelet_filters.npz
# ---------------------------------------------------------------------------

_FILTER_CACHE = {}


def get_wavelet_filters(wave_name="db4", data_dir="data"):
    """Load wavelet filter coefficients from data/wavelet_filters.npz."""
    if wave_name in _FILTER_CACHE:
        return _FILTER_CACHE[wave_name]

    import os
    path = os.path.join(data_dir, "wavelet_filters.npz")
    filters = np.load(path)
    keys = [f"{wave_name}_{s}" for s in ("dec_lo", "dec_hi", "rec_lo", "rec_hi")]
    if keys[0] not in filters:
        raise ValueError(f"Unknown wavelet '{wave_name}'.")
    result = tuple(filters[k] for k in keys)
    _FILTER_CACHE[wave_name] = result
    return result


# ---------------------------------------------------------------------------
# 1D DWT / IDWT primitives (matching pywt 'zero' mode convention)
# ---------------------------------------------------------------------------

def _dwt1d(signal, dec_lo, dec_hi):
    """Single-level 1D DWT: convolve + downsample by 2."""
    filt_len = len(dec_lo)
    n = len(signal)
    out_len = (n + filt_len - 1) // 2
    lo = convolve(signal, dec_lo, mode="full")
    hi = convolve(signal, dec_hi, mode="full")
    return lo[1::2][:out_len], hi[1::2][:out_len]


def _idwt1d(approx, detail, rec_lo, rec_hi, output_len):
    """Single-level 1D inverse DWT: upsample + convolve."""
    n = len(approx)
    filt_len = len(rec_lo)
    up_a = np.zeros(2 * n, dtype=approx.dtype)
    up_a[1::2] = approx
    up_d = np.zeros(2 * n, dtype=detail.dtype)
    up_d[1::2] = detail
    lo = convolve(up_a, rec_lo, mode="full")
    hi = convolve(up_d, rec_hi, mode="full")
    start = filt_len - 1
    return (lo + hi)[start:start + output_len]


# ---------------------------------------------------------------------------
# 2D DWT / IDWT (separable: rows then columns)
# ---------------------------------------------------------------------------

def _dwt2d(image, dec_lo, dec_hi):
    """Single-level 2D DWT → (LL, LH, HL, HH)."""
    H, W = image.shape
    filt_len = len(dec_lo)
    out_w = (W + filt_len - 1) // 2
    out_h = (H + filt_len - 1) // 2

    L_rows = np.zeros((H, out_w), dtype=image.dtype)
    H_rows = np.zeros((H, out_w), dtype=image.dtype)
    for i in range(H):
        L_rows[i], H_rows[i] = _dwt1d(image[i], dec_lo, dec_hi)

    LL = np.zeros((out_h, out_w), dtype=image.dtype)
    LH = np.zeros((out_h, out_w), dtype=image.dtype)
    HL = np.zeros((out_h, out_w), dtype=image.dtype)
    HH = np.zeros((out_h, out_w), dtype=image.dtype)
    for j in range(out_w):
        LL[:, j], LH[:, j] = _dwt1d(L_rows[:, j], dec_lo, dec_hi)
        HL[:, j], HH[:, j] = _dwt1d(H_rows[:, j], dec_lo, dec_hi)
    return LL, LH, HL, HH


def _idwt2d(LL, LH, HL, HH, rec_lo, rec_hi, output_shape):
    """Single-level 2D inverse DWT."""
    H, W = output_shape
    n_cols = LL.shape[1]
    L_rows = np.zeros((H, n_cols), dtype=LL.dtype)
    H_rows = np.zeros((H, n_cols), dtype=HL.dtype)
    for j in range(n_cols):
        L_rows[:, j] = _idwt1d(LL[:, j], LH[:, j], rec_lo, rec_hi, H)
        H_rows[:, j] = _idwt1d(HL[:, j], HH[:, j], rec_lo, rec_hi, H)
    image = np.zeros((H, W), dtype=LL.dtype)
    for i in range(H):
        image[i] = _idwt1d(L_rows[i], H_rows[i], rec_lo, rec_hi, W)
    return image


# ---------------------------------------------------------------------------
# Multi-level 2D wavelet transform with packing
# ---------------------------------------------------------------------------

def wavelet_forward(image, wave_name="db4", level=None, data_dir="data"):
    """
    Multi-level 2D forward wavelet transform.

    Parameters
    ----------
    image : (H, W) complex or float
    wave_name : str
    level : int or None
    data_dir : str

    Returns
    -------
    coeffs_array : ndarray — packed coefficients
    coeff_info : list of dict — metadata for unpacking
    original_shape : tuple
    """
    dec_lo, dec_hi, _, _ = get_wavelet_filters(wave_name, data_dir)

    H, W = image.shape
    if level is None:
        filt_len = len(dec_lo)
        level = max(1, int(np.floor(np.log2(min(H, W) / (filt_len - 1)))))

    approx = image
    details = []
    input_shapes = [image.shape]

    for _ in range(level):
        LL, LH, HL, HH = _dwt2d(approx, dec_lo, dec_hi)
        details.append({"LH": LH, "HL": HL, "HH": HH})
        input_shapes.append(approx.shape)
        approx = LL

    all_coeffs = [approx.ravel()]
    coeff_info = [{"type": "approx", "shape": approx.shape,
                   "n_levels": level, "input_shapes": input_shapes}]
    for i in range(level - 1, -1, -1):
        for key in ["LH", "HL", "HH"]:
            band = details[i][key]
            all_coeffs.append(band.ravel())
            coeff_info.append({"type": key, "shape": band.shape, "level": i})

    return np.concatenate(all_coeffs), coeff_info, image.shape


def wavelet_inverse(coeffs_array, coeff_info, original_shape,
                    wave_name="db4", data_dir="data"):
    """
    Multi-level 2D inverse wavelet transform.

    Parameters
    ----------
    coeffs_array : ndarray
    coeff_info : list of dict
    original_shape : tuple
    wave_name : str
    data_dir : str

    Returns
    -------
    image : ndarray, original_shape
    """
    _, _, rec_lo, rec_hi = get_wavelet_filters(wave_name, data_dir)

    offset = 0
    approx_meta = coeff_info[0]
    approx_size = int(np.prod(approx_meta["shape"]))
    approx = coeffs_array[offset:offset + approx_size].reshape(approx_meta["shape"])
    offset += approx_size
    n_levels = approx_meta["n_levels"]
    input_shapes = approx_meta["input_shapes"]

    detail_bands = {}
    for info in coeff_info[1:]:
        lvl = info["level"]
        size = int(np.prod(info["shape"]))
        band = coeffs_array[offset:offset + size].reshape(info["shape"])
        offset += size
        if lvl not in detail_bands:
            detail_bands[lvl] = {}
        detail_bands[lvl][info["type"]] = band

    current = approx
    for lvl in range(n_levels - 1, -1, -1):
        target_shape = input_shapes[lvl + 1]
        current = _idwt2d(current, detail_bands[lvl]["LH"],
                          detail_bands[lvl]["HL"], detail_bands[lvl]["HH"],
                          rec_lo, rec_hi, target_shape)

    return current[:original_shape[0], :original_shape[1]]


# ---------------------------------------------------------------------------
# Soft thresholding (extracted from sigpy/thresh.py)
# ---------------------------------------------------------------------------

def soft_thresh(lamda, x):
    """
    Complex soft thresholding: (|x| - lambda)_+ * sign(x).

    Parameters
    ----------
    lamda : float
    x : ndarray (can be complex)

    Returns
    -------
    ndarray
    """
    abs_x = np.abs(x)
    sign = np.where(abs_x > 0, x / abs_x, 0)
    mag = np.maximum(abs_x - lamda, 0)
    return mag * sign


# ---------------------------------------------------------------------------
# SENSE-NUFFT operators (extracted from sigpy/mri/linop.py Sense)
# ---------------------------------------------------------------------------

def sense_nufft_forward(image, coil_maps, coord):
    """
    SENSE-NUFFT forward: image -> multi-coil non-Cartesian k-space.

    A(x) = F_NU(S * x) for each coil.

    Parameters
    ----------
    image : (H, W) complex
    coil_maps : (C, H, W) complex
    coord : (M, 2) float

    Returns
    -------
    kdata : (C, M) complex
    """
    return multicoil_nufft_forward(image, coil_maps, coord)


def sense_nufft_adjoint(kdata, coil_maps, coord, image_shape):
    """
    SENSE-NUFFT adjoint: multi-coil k-space -> image.

    A^H(y) = sum_c conj(S_c) * F_NU^H(y_c)

    Parameters
    ----------
    kdata : (C, M) complex
    coil_maps : (C, H, W) complex
    coord : (M, 2) float
    image_shape : tuple (H, W)

    Returns
    -------
    image : (H, W) complex
    """
    n_coils = coil_maps.shape[0]
    result = np.zeros(image_shape, dtype=np.complex128)
    for c in range(n_coils):
        result += np.conj(coil_maps[c]) * nufft_adjoint(kdata[c], coord, image_shape)
    return result


def sense_nufft_normal(image, coil_maps, coord):
    """
    SENSE-NUFFT normal operator: A^H A (x).

    Parameters
    ----------
    image : (H, W) complex
    coil_maps : (C, H, W) complex
    coord : (M, 2) float

    Returns
    -------
    (H, W) complex
    """
    kdata = sense_nufft_forward(image, coil_maps, coord)
    return sense_nufft_adjoint(kdata, coil_maps, coord, image.shape)


# ---------------------------------------------------------------------------
# Power iteration for step size (extracted from sigpy/app.py MaxEig)
# ---------------------------------------------------------------------------

def estimate_max_eigenvalue(coil_maps, coord, max_iter=10):
    """
    Estimate max eigenvalue of A^H A via power iteration.

    Parameters
    ----------
    coil_maps : (C, H, W) complex
    coord : (M, 2) float
    max_iter : int

    Returns
    -------
    max_eig : float
    """
    image_shape = coil_maps.shape[1:]
    x = np.random.randn(*image_shape).astype(np.complex128)
    x /= np.linalg.norm(x)

    for _ in range(max_iter):
        y = sense_nufft_normal(x, coil_maps, coord)
        max_eig = np.linalg.norm(y)
        if max_eig > 0:
            x = y / max_eig
        else:
            break
    return float(max_eig)


# ---------------------------------------------------------------------------
# FISTA solver (extracted from sigpy/alg.py GradientMethod)
# ---------------------------------------------------------------------------

def fista_l1_wavelet_nufft(
    kdata, coord, coil_maps,
    lamda=5e-5,
    wave_name="db4",
    max_iter=100,
    accelerate=True,
    power_iter=10,
):
    """
    FISTA for L1-wavelet CS-MRI with non-Cartesian (NUFFT) encoding.

    Solves:
        min_x (1/2)||A x - y||^2 + lambda ||W x||_1

    where A = F_NU * S and W is the wavelet transform.

    Parameters
    ----------
    kdata : (C, M) complex
    coord : (M, 2) float
    coil_maps : (C, H, W) complex
    lamda : float
    wave_name : str
    max_iter : int
    accelerate : bool
    power_iter : int

    Returns
    -------
    x : (H, W) complex
    """
    image_shape = coil_maps.shape[1:]

    # A^H y
    AHy = sense_nufft_adjoint(kdata, coil_maps, coord, image_shape)

    # Step size via power iteration
    alpha = 1.0 / estimate_max_eigenvalue(coil_maps, coord, max_iter=power_iter)

    # Wavelet shape info (compute once)
    dummy_coeffs, coeff_info, orig_shape = wavelet_forward(
        np.zeros(image_shape, dtype=np.complex128), wave_name,
    )

    # Initialize
    x = np.zeros(image_shape, dtype=np.complex128)
    if accelerate:
        z = x.copy()
        t = 1.0

    for _ in range(max_iter):
        x_old = x.copy()

        x_input = z if accelerate else x

        # Gradient step
        gradf = sense_nufft_normal(x_input, coil_maps, coord) - AHy
        x = x_input - alpha * gradf

        # Proximal: soft-threshold in wavelet domain
        coeffs, _, _ = wavelet_forward(x, wave_name)
        coeffs = soft_thresh(lamda * alpha, coeffs)
        x = wavelet_inverse(coeffs, coeff_info, image_shape, wave_name)

        # Nesterov momentum
        if accelerate:
            t_old = t
            t = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
            z = x + ((t_old - 1) / t) * (x - x_old)

    return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def l1wav_reconstruct_single(
    kdata, coord, coil_maps, lamda=5e-5, max_iter=100,
):
    """
    Reconstruct a single non-Cartesian MRI image using L1-wavelet CS.

    Parameters
    ----------
    kdata : (C, M) complex
    coord : (M, 2) float
    coil_maps : (C, H, W) complex
    lamda : float
    max_iter : int

    Returns
    -------
    recon : (H, W) complex
    """
    return fista_l1_wavelet_nufft(
        kdata, coord, coil_maps,
        lamda=lamda, max_iter=max_iter,
    )


def l1wav_reconstruct_batch(
    kdata, coord, coil_maps, lamda=5e-5, max_iter=100,
):
    """
    Reconstruct a batch of non-Cartesian MRI images using L1-wavelet CS.

    Parameters
    ----------
    kdata : (N, C, M) complex
    coord : (N, M, 2) float
    coil_maps : (N, C, H, W) complex
    lamda : float
    max_iter : int

    Returns
    -------
    recons : (N, H, W) complex
    """
    n_samples = kdata.shape[0]
    recons = []
    for i in range(n_samples):
        recon = l1wav_reconstruct_single(
            kdata[i], coord[i], coil_maps[i], lamda, max_iter,
        )
        recons.append(recon)
    return np.stack(recons, axis=0)
