"""Explicit acquisition, linear coherence convolution and projection mathematics.

Adapted from cohere, Copyright 2019 UChicago Argonne, LLC. Full copyright,
conditions and disclaimer are retained in test/validation.md. The corrected
partial-coherence argument order and independent check are documented there.
"""
import numpy as np
from scipy import fft, ndimage


def reciprocal_wavefield(obj):
    """Centered unscaled positive-sign DFT, valid for odd and even dimensions."""
    return np.fft.fftshift(fft.ifftn(np.fft.ifftshift(obj), norm="forward"))


def direct_wavefield(wavefield):
    """Inverse of reciprocal_wavefield: negative-sign DFT divided by N."""
    return np.fft.fftshift(fft.fftn(np.fft.ifftshift(wavefield), norm="forward"))


def full_convolution(array, kernel):
    """Full zero-extended real n-D convolution, before any output-window crop."""
    shape = tuple(a + b - 1 for a, b in zip(array.shape, kernel.shape))
    padded = tuple(fft.next_fast_len(n, real=True) for n in shape)
    product = fft.rfftn(array, padded) * fft.rfftn(kernel, padded)
    return fft.irfftn(product, padded)[tuple(slice(0, n) for n in shape)]


def linear_convolution(array, kernel):
    """Zero-extended n-D convolution cropped to array.shape, SciPy 'same'."""
    full = full_convolution(array, kernel)
    start = tuple((n - 1) // 2 for n in kernel.shape)
    return full[tuple(slice(s, s + n) for s, n in zip(start, array.shape))]


def kernel_adjoint(residual, coherent, kernel_shape):
    """Exact kernel transpose of the finite same-convolution observation window.

    Forward crop begins at floor((K-1)/2). Its transpose starts at R-1 minus
    that origin in full convolution with the reversed R-shaped coherent array.
    An even K therefore needs a different crop than another 'same' convolution.
    """
    origin = tuple((n - 1) // 2 for n in kernel_shape)
    start = tuple(n - 1 - o for n, o in zip(coherent.shape, origin))
    full = full_convolution(residual, np.flip(coherent))
    return full[tuple(slice(s, s + n) for s, n in zip(start, kernel_shape))]


def amplitude_projection(wavefield, amplitude, kernel=None):
    """Return modified wavefield and squared relative amplitude residual.

    Arrays are in centered detector order. The absolute 1e-9 cutoff is
    preserved from the reference code.
    """
    absolute = np.abs(wavefield)
    converged = absolute if kernel is None else np.sqrt(
        np.maximum(linear_convolution(absolute ** 2, kernel), 0.))
    ratio = np.divide(amplitude, converged, out=np.zeros_like(converged),
                      where=converged > 1e-9)
    residual = np.where(converged != 0, converged - amplitude, 0.)
    error = float(np.sum(residual ** 2) / np.sum(amplitude ** 2))
    return wavefield * ratio, error


def crop_center(array, shape):
    """Crop by floor((old-new)/2), including the odd/even convention."""
    return array[tuple(slice((a - b) // 2, (a - b) // 2 + b)
                       for a, b in zip(array.shape, shape))]


def update_coherence(amplitude, previous, measured_roi, kernel, iterations=20):
    """Richardson-Lucy kernel update for measured = coherent * kernel.

    The extrapolated coherent ROI is normalized to the measured ROI photon
    sum, as in cohere. Finite-window kernel adjoint and sensitivity, the physical
    direction of the observation equation, and Fourier ordering are corrected.
    This is an explicit local kernel update, not a domain-library solver.
    """
    roi = crop_center(amplitude, kernel.shape)
    coherent = (2 * roi - previous) ** 2
    normalization = np.sum(measured_roi ** 2) / np.sum(coherent)
    coherent = np.sqrt(coherent * normalization) ** 2
    measured = measured_roi ** 2
    sensitivity = kernel_adjoint(np.ones_like(measured, dtype=np.float64), coherent, kernel.shape)
    sensitivity = np.where(sensitivity > 0., sensitivity, 1.)
    result = kernel.copy()
    for _ in range(iterations):
        denominator = linear_convolution(coherent, result)
        denominator = np.maximum(denominator, 1e-30)
        result = result * kernel_adjoint(measured / denominator, coherent, kernel.shape) / sensitivity
        result = np.maximum(result, 0.)
    result = np.abs(result.real)
    return result / result.sum()


def shrink_support(obj, sigma, threshold=0.1):
    """Gaussian |object|, half-sample reflect boundary, truncate=4, >= cutoff."""
    blurred = ndimage.gaussian_filter(np.abs(obj), sigma=sigma)
    return (blurred >= threshold * blurred.max()).astype(np.int64)


def observation_intensity(obj, kernel):
    """Predicted detector counts from centered coherent intensity and linear blur."""
    intensity = linear_convolution(np.abs(reciprocal_wavefield(obj)) ** 2, kernel)
    return np.maximum(intensity, 0.)


def raw_observation_intensity(obj, kernel, peak_shift, padding=None):
    """Undo a fixed observed-peak roll after centered linear intensity blur.

    peak_shift is determined from the public observation's first maximum,
    never from a hidden object. Convolution is evaluated before undoing the
    roll: finite linear convolution and cyclic rolling do not commute.
    """
    intensity = observation_intensity(obj, kernel)
    shift = np.asarray(peak_shift, dtype=np.int64)
    if shift.shape != (intensity.ndim,):
        raise ValueError('peak_shift must have one index displacement per axis')
    intensity = np.roll(intensity, -shift, axis=tuple(range(intensity.ndim)))
    if padding is not None:
        if len(padding) != intensity.ndim:
            raise ValueError('padding must contain one pair per axis')
        intensity = intensity[tuple(slice(left, n-right if right else None)
                                    for n, (left, right) in zip(intensity.shape, padding))]
    return intensity


def projected_strain(obj, basis_nm, q_vector, support_fraction=np.exp(-1)):
    """Return relative projected strain and valid interior mask on oblique grid.

    Wrapped nearest-neighbor complex ratios avoid an arbitrary phase unwrap.
    Strain is qhat dot grad(arg(object))/|q|; a support-weighted constant mean
    is removed. Undefined exterior/one-voxel boundary is set to zero. No
    refraction correction or full strain tensor is inferred.
    """
    mask = np.abs(obj) >= support_fraction * np.abs(obj).max()
    valid = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3)))
    grad_index = np.stack([np.angle(np.roll(obj, -1, axis) *
                                  np.conj(np.roll(obj, 1, axis))) / 2
                           for axis in range(3)])
    grad_lab = np.einsum("ij,j...->i...", np.linalg.inv(basis_nm).T, grad_index)
    q = np.asarray(q_vector)
    strain = np.einsum("i,i...->...", q, grad_lab) / np.dot(q, q)
    if valid.any():
        strain -= strain[valid].mean()
    return np.where(valid, strain, 0.), valid
