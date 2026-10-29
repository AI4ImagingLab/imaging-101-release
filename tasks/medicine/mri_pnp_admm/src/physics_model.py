"""
Single-Coil Cartesian CS-MRI Forward Model
=============================================

Implements the forward model for compressed sensing MRI:

    y = M * F(x) + noise

where:
    x : 2D image (m, n) in spatial domain
    F : 2D discrete Fourier transform
    M : binary k-space undersampling mask
    y : observed (undersampled, noisy) k-space measurements

The inverse problem is to recover x from y given M and noise statistics.

The adjoint (zero-filled reconstruction) is:

    x_zf = |F^{-1}(y)|

which serves as the initialization for iterative methods.

Reference
---------
Ryu et al., "Plug-and-Play Methods Provably Converge with Properly
Trained Denoisers," ICML 2019.
"""

import numpy as np


def forward_model(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply the CS-MRI forward model: undersampled Fourier encoding.

    Parameters
    ----------
    image : ndarray, (m, n) float64
        Ground truth image in spatial domain.
    mask : ndarray, (m, n) float64
        Binary k-space undersampling mask.

    Returns
    -------
    kspace : ndarray, (m, n) complex128
        Undersampled k-space (clean, no noise).
    """
    return np.fft.fft2(image) * mask


def add_noise(kspace: np.ndarray, noises: np.ndarray) -> np.ndarray:
    """
    Add complex Gaussian noise to k-space measurements.

    Parameters
    ----------
    kspace : ndarray, (m, n) complex128
        Clean undersampled k-space.
    noises : ndarray, (m, n) complex128
        Complex noise to add.

    Returns
    -------
    noisy_kspace : ndarray, (m, n) complex128
        Noisy k-space measurements.
    """
    return kspace + noises


def simulate_observation(image: np.ndarray, mask: np.ndarray,
                         noises: np.ndarray) -> np.ndarray:
    """
    Simulate the full CS-MRI measurement process.

    Parameters
    ----------
    image : ndarray, (m, n) float64
        Ground truth image.
    mask : ndarray, (m, n) float64
        Binary undersampling mask.
    noises : ndarray, (m, n) complex128
        Measurement noise.

    Returns
    -------
    y : ndarray, (m, n) complex128
        Observed noisy undersampled k-space.
    """
    return forward_model(image, mask) + noises


def zero_filled_recon(y: np.ndarray) -> np.ndarray:
    """
    Zero-filled reconstruction (adjoint operation).

    Parameters
    ----------
    y : ndarray, (m, n) complex128
        Observed k-space data.

    Returns
    -------
    x_zf : ndarray, (m, n) float64
        Magnitude of inverse FFT (zero-filled reconstruction).
    """
    return np.abs(np.fft.ifft2(y))


def data_fidelity_proximal(vtilde: np.ndarray, y: np.ndarray,
                           mask: np.ndarray, alpha: float) -> np.ndarray:
    """
    Proximal operator for the data fidelity term ||M*F(x) - y||^2 / (2*alpha).

    Solved in closed form in Fourier domain: for sampled indices,
    vf[i] = (La2 * vf[i] + y[i]) / (1 + La2) where La2 = 1/(2*alpha).
    For unsampled indices, vf[i] is unchanged.

    Parameters
    ----------
    vtilde : ndarray, (m, n) float64
        Input to proximal operator (x + u in ADMM).
    y : ndarray, (m, n) complex128
        Observed k-space measurements.
    mask : ndarray, (m, n) float64
        Binary undersampling mask.
    alpha : float
        ADMM penalty parameter.

    Returns
    -------
    v : ndarray, (m, n) float64
        Result of proximal step (real part of IFFT).
    """
    index = np.nonzero(mask)
    vf = np.fft.fft2(vtilde)
    La2 = 1.0 / (2.0 * alpha)
    vf[index] = (La2 * vf[index] + y[index]) / (1.0 + La2)
    return np.real(np.fft.ifft2(vf))
