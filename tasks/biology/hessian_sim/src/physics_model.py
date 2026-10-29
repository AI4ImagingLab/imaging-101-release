"""Forward imaging model for 2-beam structured illumination microscopy (SIM).

The sample fluorophore density ``x(r)`` is multiplied by a sinusoidal
illumination pattern and then imaged through a diffraction-limited optical
system described by its point spread function ``h(r)``:

    y_{a,p}(r) = [ (1 + m cos(2π k_a · r + φ_{a,p})) x(r) ] * h(r) + n(r)

- ``k_a``       : pattern wavevector for angle ``a``
- ``φ_{a,p}``   : phase for angle ``a``, phase step ``p``
- ``m``         : modulation depth (≤ 1)
- ``h(r)``      : point spread function, whose Fourier transform is the OTF
- ``n(r)``      : measurement noise

This module provides the synthetic OTF used when no measured OTF is
available, together with helpers for centred 2D FFT convolution and the
Fourier shift theorem used throughout the SIM pipeline.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def generate_otf(n: int, na: float, wavelength: float, pixel_size: float) -> np.ndarray:
    """Synthetic cone-shaped 2D optical transfer function.

    Parameters
    ----------
    n : size of the (n, n) OTF grid in pixels.
    na : numerical aperture.
    wavelength : emission wavelength in nm.
    pixel_size : pixel size in nm.

    Returns
    -------
    otf : ``(n, n)`` float64 array, normalised to max = 1.
    """
    cutoff_freq = 2 * na / (wavelength * 1e-3)        # 1/um
    freq_per_pixel = 1.0 / (n * pixel_size * 1e-3)    # 1/um
    fc_pixels = cutoff_freq / freq_per_pixel

    ky, kx = np.mgrid[-n // 2:n // 2, -n // 2:n // 2]
    kr = np.sqrt(kx ** 2 + ky ** 2)

    otf = np.clip(1.0 - kr / fc_pixels, 0, 1)
    otf = otf / otf.max()
    return otf.astype(np.float64)


def pad_to_size(arr: np.ndarray, target_shape) -> np.ndarray:
    """Centre zero-pad a 2D array to ``target_shape``."""
    sy, sx = arr.shape
    ty, tx = target_shape
    py = (ty - sy) // 2
    px = (tx - sx) // 2
    out = np.zeros(target_shape, dtype=arr.dtype)
    out[py:py + sy, px:px + sx] = arr
    return out


def dft_conv(h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """FFT-based linear convolution using centred spectra."""
    K = np.array(h.shape)
    L = np.array(g.shape)
    N = K + L
    H = fftshift(fft2(ifftshift(h), s=tuple(N)))
    G = fftshift(fft2(ifftshift(g), s=tuple(N)))
    result = fftshift(ifft2(ifftshift(H * G)))
    out_size = K + L - 1
    start = ((N - out_size) // 2).astype(int)
    return result[start[0]:start[0] + out_size[0], start[1]:start[1] + out_size[1]]


def shift_otf(H_2n: np.ndarray, kx: float, ky: float, n: int) -> np.ndarray:
    """Shift an OTF in Fourier space via the Fourier shift theorem.

    Equivalent to convolving by a translation kernel — used to move
    sideband orders back to their true location in the super-resolved
    reconstruction.
    """
    xx, yy = np.meshgrid(np.arange(2 * n), np.arange(2 * n))
    Ir = np.exp(1j * (kx * xx + ky * yy))
    h_real = fftshift(ifft2(ifftshift(H_2n)))
    return fftshift(fft2(ifftshift(h_real * Ir)))
