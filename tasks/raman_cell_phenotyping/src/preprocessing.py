"""
Preprocessing pipeline for volumetric Raman spectroscopy data.

Implements spectral cropping, cosmic-ray removal (Whitaker-Hayes despiking),
Savitzky-Golay denoising, asymmetric least-squares baseline correction, and
min-max normalisation.
"""

import json

import numpy as np
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_observation(data_dir: str = "data") -> dict:
    """Load raw_data.npz.

    Returns
    -------
    dict with keys:
        spectral_volume : ndarray, shape (40, 40, 10, B)
        spectral_axis   : ndarray, shape (B,)
    """
    raw = np.load(f"{data_dir}/raw_data.npz")
    return {
        "spectral_volume": raw["spectral_volume"][0],  # drop batch dim
        "spectral_axis": raw["spectral_axis"],
    }


def load_metadata(data_dir: str = "data") -> dict:
    """Load meta_data.json."""
    with open(f"{data_dir}/meta_data.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Individual preprocessing steps
# ---------------------------------------------------------------------------

def crop(intensity_data: np.ndarray, spectral_axis: np.ndarray,
         region: tuple) -> tuple:
    """Crop spectra to a wavenumber region.

    Parameters
    ----------
    intensity_data : ndarray, shape (..., B)
    spectral_axis  : ndarray, shape (B,)
    region         : (start_cm1, end_cm1)

    Returns
    -------
    (cropped_data, cropped_axis)
    """
    lo, hi = region
    if lo > hi:
        lo, hi = hi, lo
    mask = (spectral_axis >= lo) & (spectral_axis <= hi)
    return intensity_data[..., mask], spectral_axis[mask]


def _modified_z_score(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


def _despike_spectrum(spectrum: np.ndarray, kernel_size: int,
                      threshold: float) -> np.ndarray:
    """Remove cosmic-ray spikes from a single spectrum (Whitaker-Hayes)."""
    out = spectrum.copy()
    z_scores = np.abs(_modified_z_score(np.diff(out)))
    spikes = z_scores > threshold

    while np.any(spikes):
        changed = False
        for i in range(len(spikes)):
            if not spikes[i]:
                continue
            lo = max(0, i - kernel_size)
            hi = min(len(out) - 1, i + 1 + kernel_size)
            neighbours = np.arange(lo, hi)
            good = neighbours[~spikes[np.clip(neighbours, 0, len(spikes) - 1)]]
            if len(good) == 0:
                continue
            out[i] = np.mean(out[good])
            spikes[i] = False
            changed = True
        if not changed:
            break
    return out


def despike(intensity_data: np.ndarray, spectral_axis: np.ndarray,
            kernel_size: int = 3, threshold: float = 8) -> tuple:
    """Whitaker-Hayes cosmic-ray despiking.

    Parameters
    ----------
    intensity_data : ndarray, shape (..., B)
    kernel_size    : int, neighbourhood radius
    threshold      : float, modified z-score threshold

    Returns
    -------
    (despiked_data, spectral_axis)
    """
    result = np.apply_along_axis(
        _despike_spectrum, -1, intensity_data,
        kernel_size=kernel_size, threshold=threshold,
    )
    return result, spectral_axis


def denoise_savgol(intensity_data: np.ndarray, spectral_axis: np.ndarray,
                   window_length: int = 7, polyorder: int = 3) -> tuple:
    """Savitzky-Golay spectral smoothing.

    Parameters
    ----------
    intensity_data : ndarray, shape (..., B)
    window_length  : int, odd
    polyorder      : int, < window_length

    Returns
    -------
    (smoothed_data, spectral_axis)
    """
    smoothed = savgol_filter(intensity_data, window_length, polyorder, axis=-1)
    return smoothed, spectral_axis


def _asls_single(spectrum: np.ndarray, lam: float, p: float,
                 max_iter: int, tol: float) -> np.ndarray:
    """Asymmetric Least Squares baseline for a single spectrum.

    Solves iteratively:
        (W + lam * D^T D) z = W y
    where W = diag(w), D is the 2nd-order finite difference matrix,
    and weights are updated asymmetrically:
        w_i = p   if y_i > z_i   (point above baseline)
        w_i = 1-p if y_i <= z_i  (point below baseline)

    References
    ----------
    Eilers & Boelens, Baseline correction with asymmetric least squares
    smoothing, 2005.
    """
    m = len(spectrum)
    # 2nd-order difference matrix
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(m - 2, m), format="csc")
    DTD = lam * D.T.dot(D)
    w = np.ones(m)
    z = spectrum.copy()
    for _ in range(max_iter):
        W = sparse.diags(w, 0, shape=(m, m), format="csc")
        z_new = spsolve(W + DTD, w * spectrum)
        if np.linalg.norm(z_new - z) / (np.linalg.norm(z) + 1e-30) < tol:
            z = z_new
            break
        z = z_new
        w = np.where(spectrum > z, p, 1 - p)
    return z


def baseline_asls(intensity_data: np.ndarray, spectral_axis: np.ndarray,
                  lam: float = 1e6, p: float = 1e-2,
                  max_iter: int = 50, tol: float = 1e-3) -> tuple:
    """Asymmetric least-squares baseline correction.

    Parameters
    ----------
    intensity_data : ndarray, shape (..., B)
    lam            : float, smoothing parameter
    p              : float, asymmetry parameter
    max_iter       : int
    tol            : float

    Returns
    -------
    (corrected_data, spectral_axis)
    """
    baseline = np.apply_along_axis(
        _asls_single, -1, intensity_data,
        lam=lam, p=p, max_iter=max_iter, tol=tol,
    )
    return intensity_data - baseline, spectral_axis


def normalise_minmax(intensity_data: np.ndarray, spectral_axis: np.ndarray,
                     pixelwise: bool = False) -> tuple:
    """Min-max normalisation to [0, 1].

    Parameters
    ----------
    intensity_data : ndarray, shape (..., B)
    pixelwise      : bool
        If False (default), use global min/max across all spectra.

    Returns
    -------
    (normalised_data, spectral_axis)
    """
    if pixelwise:
        mins = intensity_data.min(axis=-1, keepdims=True)
        maxs = intensity_data.max(axis=-1, keepdims=True)
    else:
        mins = intensity_data.min()
        maxs = intensity_data.max()
    denom = maxs - mins
    if np.isscalar(denom):
        denom = max(denom, 1e-30)
    else:
        denom = np.maximum(denom, 1e-30)
    return (intensity_data - mins) / denom, spectral_axis


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_volume(spectral_volume: np.ndarray,
                      spectral_axis: np.ndarray) -> tuple:
    """Apply the full preprocessing pipeline.

    Steps:
        1. Crop to fingerprint region (700-1800 cm-1)
        2. Whitaker-Hayes despiking
        3. Savitzky-Golay smoothing (window=7, polyorder=3)
        4. ASLS baseline correction
        5. Global min-max normalisation

    Parameters
    ----------
    spectral_volume : ndarray, shape (X, Y, Z, B)
    spectral_axis   : ndarray, shape (B,)

    Returns
    -------
    (processed_volume, processed_axis)
    """
    data, axis = crop(spectral_volume, spectral_axis, region=(700, 1800))
    data, axis = despike(data, axis)
    data, axis = denoise_savgol(data, axis, window_length=7, polyorder=3)
    data, axis = baseline_asls(data, axis)
    data, axis = normalise_minmax(data, axis, pixelwise=False)
    return data, axis
