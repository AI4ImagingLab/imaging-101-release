"""Explicit circular blur and five-point Laplacian operators.

Extracted from SDeconv 1.0.4, copyright (c) 2021 STracking, BSD-3-Clause.
"""
import numpy as np

def transfer_function(kernel: np.ndarray) -> np.ndarray:
    shifts = tuple(int(-n / 2) for n in kernel.shape)
    return np.fft.fft2(np.roll(kernel, shifts, axis=(0, 1)))

def laplacian_kernel(shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float64)
    y, x = (n // 2 for n in shape)
    out[y, x] = 4
    out[y-1, x] = out[y+1, x] = out[y, x-1] = out[y, x+1] = -1
    return out

def apply_blur(image: np.ndarray, otf: np.ndarray, adjoint: bool = False) -> np.ndarray:
    multiplier = otf.conj() if adjoint else otf
    return np.fft.ifft2(np.fft.fft2(image) * multiplier).real

def laplacian(image: np.ndarray) -> np.ndarray:
    return 4 * image - sum(np.roll(image, shift, axis) for shift, axis in [(-1, 0), (1, 0), (-1, 1), (1, 1)])
