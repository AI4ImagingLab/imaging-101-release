# Copyright 2007-2023 The HyperSpy developers; adaptations 2026 imaging-101.
# Adapted from HoloSpy v0.3, holospy/reconstruct.py. GPL-3.0-or-later; see test/validation.md (full license record).
"""Explicit off-axis interference and reciprocal-space demodulation operators."""
import numpy as np
from .preprocessing import PIXEL_SIZE_NM


def radial_frequency(shape: tuple[int, int], pixel_size: float = PIXEL_SIZE_NM) -> np.ndarray:
    """Return unshifted DFT radial frequencies in inverse nm."""
    fy = np.fft.fftfreq(shape[0], pixel_size)
    fx = np.fft.fftfreq(shape[1], pixel_size)
    return np.hypot(fy[:, None], fx[None, :])


def aperture(r: np.ndarray, radius: float, smoothness: float) -> np.ndarray:
    """Tanh aperture; radius and smoothness use the same units as r."""
    if radius <= 0 or smoothness <= 0:
        raise ValueError('Aperture radius and smoothness must be positive')
    return .5 * (1 - np.tanh((np.abs(r) - radius) / (.5 * smoothness)))


def sideband_wave(image: np.ndarray, position: tuple[int, int], radius: float,
                  smoothness: float | None = None) -> np.ndarray:
    """Demodulate the negative carrier sideband by positive roll; no output crop."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('Expected a finite real 2D intensity')
    smoothness = radius * .05 if smoothness is None else smoothness
    df = np.mean(1 / (np.array(image.shape) * PIXEL_SIZE_NM))
    window = aperture(radial_frequency(image.shape), radius * df, smoothness * df)
    spectrum = np.fft.fft2(image) / image.size
    centered = np.roll(spectrum, tuple(position), axis=(0, 1))
    return np.fft.ifft2(centered * window) * image.size


def interference(object_wave: np.ndarray, reference_wave: np.ndarray,
                 position: tuple[int, int]) -> np.ndarray:
    """Ideal intensity |object + reference*exp(i*carrier)|^2 in amplitude^2 units.

    Positive-roll extraction selects object*conj(reference), fixing phase sign.
    """
    y, x = np.indices(object_wave.shape)
    phase = 2 * np.pi * (position[0] * y / y.shape[0] + position[1] * x / x.shape[1])
    return np.abs(object_wave + reference_wave * np.exp(1j * phase)) ** 2
