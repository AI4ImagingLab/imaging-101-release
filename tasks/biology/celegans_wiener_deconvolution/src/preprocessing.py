"""Observation loading and upstream reflection-extension convention."""
from pathlib import Path
import numpy as np

PAD = 13
SHAPE = (1, 310, 310)

def load_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        image, psf = data['observation'], data['psf']
    if image.shape != SHAPE or psf.shape != (1, 13, 13):
        raise ValueError('Invalid observation or PSF shape')
    if not np.isfinite(image).all() or not np.isfinite(psf).all():
        raise ValueError('Nonfinite input')
    if np.any(image < 0) or np.any(psf < 0) or psf.sum() <= 0:
        raise ValueError('Inputs must be nonnegative with nonzero PSF')
    return image[0].astype(np.float64), psf[0].astype(np.float64)

def prepare(image: np.ndarray, psf: np.ndarray, pad: int = PAD) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float64)
    padded = np.pad(image, pad, mode='reflect') if pad else image.copy()
    kernel = np.zeros(padded.shape, dtype=np.float64)
    starts = [int(n / 2 - k / 2) + 1 for n, k in zip(padded.shape, psf.shape)]
    kernel[starts[0]:starts[0]+psf.shape[0], starts[1]:starts[1]+psf.shape[1]] = psf / psf.sum()
    return padded, kernel
