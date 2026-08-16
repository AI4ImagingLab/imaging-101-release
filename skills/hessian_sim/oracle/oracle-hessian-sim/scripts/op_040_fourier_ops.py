"""Centred-FFT helpers: pad_to_size, dft_conv, shift_otf.
Self-contained; run from the eval run root with the run-local python.
"""
from __future__ import annotations
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def pad_to_size(arr, target_shape):
    sy, sx = arr.shape
    ty, tx = target_shape
    py = (ty - sy) // 2
    px = (tx - sx) // 2
    out = np.zeros(target_shape, dtype=arr.dtype)
    out[py:py + sy, px:px + sx] = arr
    return out


def dft_conv(h, g):
    K = np.array(h.shape)
    L = np.array(g.shape)
    N = K + L
    H = fftshift(fft2(ifftshift(h), s=tuple(N)))
    G = fftshift(fft2(ifftshift(g), s=tuple(N)))
    result = fftshift(ifft2(ifftshift(H * G)))
    out_size = K + L - 1
    start = ((N - out_size) // 2).astype(int)
    return result[start[0]:start[0] + out_size[0], start[1]:start[1] + out_size[1]]


def shift_otf(H_2n, kx, ky, n):
    xx, yy = np.meshgrid(np.arange(2 * n), np.arange(2 * n))
    Ir = np.exp(1j * (kx * xx + ky * yy))
    h_real = fftshift(ifft2(ifftshift(H_2n)))
    return fftshift(fft2(ifftshift(h_real * Ir)))


if __name__ == "__main__":
    a = np.zeros((8, 8)); a[4, 4] = 1.0
    s = shift_otf(pad_to_size(a, (16, 16)), 0.0, 0.0, 8)
    print("shift_otf ok", np.isfinite(s).all())
