"""Explicit covariance operations and incoherent image formation.

CLASS FFT scales follow the released implementation; physical image formation
uses the spatial axes. Source attribution and output fixes: test/validation.md.
"""
import numpy as np
import torch


def circular_convolution(a, b):
    """Circular matrix convolution with upstream extra 1/(N*M) scale."""
    transformed = torch.fft.fft2(a)
    transformed *= torch.fft.fft2(b)
    return torch.fft.ifft2(transformed)/transformed.numel()


def circular_correlation(a, b):
    """Circular correlation; one-column b is broadcast as in upstream."""
    transformed = torch.fft.fft2(a)
    transformed *= torch.fft.fft2(b).conj()
    return torch.fft.ifft2(transformed)/transformed.numel()


def phase_direction(matrix):
    """Complex (N,) correction without constructing an N-by-N covariance."""
    n,m = matrix.shape
    convolution = circular_convolution(matrix.conj().flipud(), matrix.fliplr())[:,m-1]
    correlation = circular_correlation(matrix.flipud(), convolution[:,None].conj())
    matched = torch.roll(correlation.flipud(), 1, 0)[:n,:m]
    return torch.mean(matrix.conj()*matched, 1)


def spatial_frames(matrix, shape):
    """Unpack row ix*H+iy, inverse spatial FFT, return complex (M,H,W)."""
    height,width = shape
    if matrix.shape[0] != height*width:
        raise ValueError('Matrix row count does not match spatial shape')
    spectra = matrix.reshape(width,height,matrix.shape[1]).permute(2,1,0)
    return torch.fft.ifft2(torch.fft.ifftshift(spectra, dim=(-2,-1)))


def estimate_mtf(matrix):
    """S16/S17: relative MTF vector sqrt(row power / peak row power)."""
    power = (matrix.abs()**2).sum(1)
    if not torch.isfinite(power).all() or float(power.max()) <= 0:
        raise ValueError('MTF requires nonzero finite observations')
    return torch.sqrt(power/power.max())


def rms_object(matrix, shape, mtf=None, sigma=0.01):
    """S18 reweight, spatial inverse, RMS; peak-normalized float32 H,W.

    mtf=None gives the phase-only comparison. The regularizer is relative
    to the peak-normalized MTF; it is applied to each frame before RMS.
    """
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    corrected = matrix if mtf is None else matrix/(mtf[:,None]+sigma)
    frames = spatial_frames(corrected, shape)
    image = torch.sqrt(torch.mean(frames.abs()**2, 0))
    peak = image.max()
    if not torch.isfinite(peak) or float(peak) <= 0:
        raise ValueError('Degenerate reconstructed object')
    return image/peak


def incoherent_forward(object_image, illumination, psf):
    """Periodic intensity model P*(O*S), NumPy arrays (...,H,W).

    A normalized nonnegative PSF conserves total intensity. Used for
    independent physical checks, not as a replacement blind solver.
    """
    return np.fft.ifft2(np.fft.fft2(np.asarray(object_image)*illumination)*np.fft.fft2(psf)).real


def covariance_coherence(matrix):
    """Coherent-diagonal energy / squared total Fourier camera power.

    sum_d |sum_a V[a+d,a]|^2 / trace(V)^2, V=R R^H, circular flattened-k
    differences. A covariance model check without an explicit N*N array.
    """
    # Camera DN and FFT scaling can overflow a float32 fourth-order moment.
    matrix = matrix/torch.sqrt(torch.sum(matrix.abs().double()**2))
    transformed = torch.fft.fft(matrix, dim=0)
    diagonals = torch.fft.ifft(torch.sum(transformed.abs()**2, dim=1))
    return float(torch.sum(diagonals.abs()**2)/(torch.sum(matrix.abs()**2)**2))
