# Copyright 2007-2023 The HyperSpy developers; adaptations 2026 imaging-101.
# HoloSpy v0.3 Fourier method, GPL-3.0-or-later; see test/validation.md (full license record).
"""Closed-form calibrated wave reconstruction, with explicit carrier estimation."""
import numpy as np
from .physics_model import radial_frequency, aperture, sideband_wave
from .preprocessing import SHAPE


def estimate_carrier(vacuum: np.ndarray) -> tuple[tuple[int, int], float]:
    """Find lower-half vacuum Fourier peak after a 5%-Nyquist center stop."""
    radial = radial_frequency(vacuum.shape)
    stop = 1 - aperture(radial, .05 * radial.max(), 1e-6)
    spectrum = np.fft.fft2(vacuum) / vacuum.size
    lower = np.abs((spectrum * stop)[:vacuum.shape[0] // 2])
    position = tuple(int(v) for v in np.unravel_index(np.argmax(lower), lower.shape))
    corners = np.array([[0, 0], [0, vacuum.shape[1]], [vacuum.shape[0], 0], vacuum.shape])
    radius = .5 * np.min(np.linalg.norm(np.array(position) - corners, axis=1))
    return position, float(radius)


def reconstruct(hologram: np.ndarray, vacuum: np.ndarray) -> np.ndarray:
    """Return complex128 calibrated transmission, shape (1,512,512), dimensionless."""
    if hologram.shape != SHAPE or vacuum.shape != SHAPE:
        raise ValueError(f'Expected observation shapes {SHAPE}')
    if not np.isfinite(hologram).all() or not np.isfinite(vacuum).all():
        raise ValueError('Nonfinite observations')
    position, radius = estimate_carrier(vacuum[0])
    obj = sideband_wave(hologram[0], position, radius)
    ref = sideband_wave(vacuum[0], position, radius)
    if np.any(np.abs(ref) <= np.finfo(float).eps * np.max(np.abs(ref))):
        raise ValueError('Vacuum wave has numerical zeros; division is undefined')
    return (obj / ref)[None]
