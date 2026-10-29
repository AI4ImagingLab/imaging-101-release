"""Laplacian-regularized inverse, evaluated directly without opaque solvers."""
import numpy as np
from .preprocessing import prepare, PAD
from .physics_model import transfer_function, laplacian_kernel

BETA = 0.005

def reconstruct(image: np.ndarray, psf: np.ndarray, beta: float = BETA, pad: int = PAD,
                return_full: bool = False) -> np.ndarray:
    if beta <= 0 or pad < 0:
        raise ValueError('beta must be positive and pad nonnegative')
    padded, kernel = prepare(image, psf, pad)
    h = transfer_function(kernel)
    ell = np.fft.fft2(laplacian_kernel(padded.shape))
    spectrum = np.fft.fft2(padded) * h.conj() / (np.abs(h)**2 + beta * np.abs(ell)**2)
    full = np.fft.ifft2(spectrum).real
    return full if return_full or pad == 0 else full[pad:-pad, pad:-pad]
