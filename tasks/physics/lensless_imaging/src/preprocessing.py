"""
Preprocessing utilities for lensless (DiffuserCam) imaging.

Loads the PSF and lensless measurement, applies downsampling, background
subtraction, and normalisation so that both arrays are in [0, 1] and
share the same spatial dimensions (H, W, C).
"""

import numpy as np
from PIL import Image
from skimage.transform import resize


def load_image(path: str, downsample: int = 1) -> np.ndarray:
    """Load an RGB image from *path*, convert to float32 in [0, 1].

    Parameters
    ----------
    path : str
        Path to a PNG / JPEG image file.
    downsample : int
        Integer downsampling factor applied via area-averaging anti-aliasing.

    Returns
    -------
    img : ndarray, shape (H, W, 3), dtype float32, values in [0, 1]
    """
    img = np.array(Image.open(path)).astype(np.float32) / 255.0
    if downsample > 1:
        h = img.shape[0] // downsample
        w = img.shape[1] // downsample
        img = resize(img, (h, w), anti_aliasing=True).astype(np.float32)
    return img


def preprocess_psf(psf: np.ndarray) -> np.ndarray:
    """Background-subtract and normalise a PSF array.

    Steps:
    1. Subtract the minimum value (removes camera dark current).
    2. Divide by the maximum value so that PSF ∈ [0, 1].

    Parameters
    ----------
    psf : ndarray, shape (H, W, C), dtype float32

    Returns
    -------
    psf_norm : ndarray, same shape, dtype float32, values in [0, 1]
    """
    psf = psf - psf.min()
    if psf.max() > 0:
        psf = psf / psf.max()
    return psf.astype(np.float32)


def preprocess_measurement(data: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Background-subtract and normalise a raw lensless measurement.

    The background level is estimated as the minimum over the PSF image
    (dark level of the sensor), which is then subtracted from the
    measurement.  The result is divided by the PSF's maximum to bring
    measurements to approximately the same scale as the PSF.

    Parameters
    ----------
    data : ndarray, shape (H, W, C), dtype float32
    psf  : ndarray, shape (H, W, C), dtype float32  (raw, before preprocess_psf)

    Returns
    -------
    data_norm : ndarray, same shape, dtype float32
    """
    data = data - psf.min()         # subtract dark-current baseline
    data = np.clip(data, 0, None)   # clamp negatives
    if psf.max() > 0:
        data = data / psf.max()     # same scale as normalised PSF
    return data.astype(np.float32)


def load_data(
    psf_path: str,
    data_path: str,
    downsample: int = 4,
) -> tuple:
    """High-level loader: reads, downsamples, and normalises PSF + measurement.

    Parameters
    ----------
    psf_path  : str   Path to PSF image (PNG).
    data_path : str   Path to raw lensless measurement image (PNG).
    downsample : int  Downsampling factor (default 4).

    Returns
    -------
    psf  : ndarray, shape (H, W, 3), float32, in [0, 1]
    data : ndarray, shape (H, W, 3), float32, in [0, 1]
    """
    psf_raw  = load_image(psf_path,  downsample=downsample)
    data_raw = load_image(data_path, downsample=downsample)

    data_norm = preprocess_measurement(data_raw, psf_raw)
    psf_norm  = preprocess_psf(psf_raw)

    return psf_norm, data_norm


def load_npz(npz_path: str) -> tuple:
    """Load preprocessed data from a .npz file.

    The npz file must contain keys 'psf' and 'measurement', each with
    shape (1, H, W, C) following the batch-first convention.

    Returns
    -------
    psf  : ndarray, shape (H, W, C)
    data : ndarray, shape (H, W, C)
    """
    npz = np.load(npz_path)
    psf  = npz["psf"][0]           # remove batch dim
    data = npz["measurement"][0]
    return psf.astype(np.float32), data.astype(np.float32)
