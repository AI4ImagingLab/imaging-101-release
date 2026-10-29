"""
Forward model for Fourier ptychography (FP).

In FP, the sample is fixed and illuminated from different angles by an LED array.
Each LED at angle θ_j shifts the object's spatial frequency spectrum by:

    q_j = 2π sin(θ_j) / λ ≈ 2π encoder_j / (λ z_led)

The microscope's pupil P̃(q) (a circle of radius NA/(λ/dxp)) bandpass-filters the
shifted spectrum. The detector records the low-resolution (LR) image:

    I_j(r) = |IFT{P̃(q) · Õ(q − q_j)}|²

where Õ = FT{O} is the object's spatial frequency spectrum.

In this way, different LEDs illuminate different regions of the object's k-space.
Overlapping coverage in k-space (from adjacent LEDs) enables phase retrieval and
super-resolution reconstruction.

The mathematical equivalence between CP and FP (Loetgering et al. 2023, appendix):
- In CP: probe P localizes real-space illumination, object scattered into k-space
- In FP: pupil P̃ localizes k-space bandwidth, object imaged in real space
- Both share the same 4D data cube; the two formulations are related by a phase-space rotation
"""

import numpy as np
from .utils import fft2c, ifft2c


def compute_pupil_mask(Nd: int, dxp: float, wavelength: float, NA: float) -> np.ndarray:
    """
    Compute the binary circular pupil (bandpass filter) in the Fourier plane.

    The pupil limits the spatial frequencies that the microscope objective
    can transmit. In k-space, the cutoff frequency is:

        k_cutoff = 2π NA / λ

    In discrete pixel units (for an Nd × Nd grid with pixel size dxp):

        r_pupil = NA / (λ / (Nd * dxp)) = NA * Nd * dxp / λ   [pixels]

    Parameters
    ----------
    Nd : int
        Pupil/detector array size.
    dxp : float
        Object pixel size [m] (= dxd / magnification).
    wavelength : float
        Illumination wavelength [m].
    NA : float
        Numerical aperture.

    Returns
    -------
    pupil : ndarray, shape (Nd, Nd), float32
        Binary pupil mask (1 inside NA circle, 0 outside).
        Returned in fftshift convention (DC at center).
    """
    fx = np.fft.fftshift(np.fft.fftfreq(Nd))  # cycles/pixel, DC at center
    FX, FY = np.meshgrid(fx, fx)
    # cutoff: f = NA * dxp / wavelength in cycles/pixel
    f_cutoff = NA * dxp / wavelength
    pupil = ((FX**2 + FY**2) <= f_cutoff**2).astype(np.float32)
    return pupil


def compute_kspace_shift(
    led_pos: np.ndarray, z_led: float, wavelength: float, Nd: int, dxp: float
) -> np.ndarray:
    """
    Compute the k-space shift (in pixels) for a given LED position.

    For a small-angle LED at (uy, ux) from the optical axis at distance z_led:
        sin(θ) ≈ encoder / z_led
        k_shift [cycles/m] = sin(θ) / λ
        k_shift [pixels] = k_shift [cycles/m] * Nd * dxp

    Parameters
    ----------
    led_pos : ndarray, shape (2,), float
        LED position (row, col) in [m] from optical axis.
    z_led : float
        LED-to-sample distance [m].
    wavelength : float
        Illumination wavelength [m].
    Nd : int
        Detector/pupil array size.
    dxp : float
        Object pixel size [m].

    Returns
    -------
    shift_px : ndarray, shape (2,), float
        k-space shift in pixels (row_shift, col_shift) in the Nd × Nd grid.
    """
    # sin(θ) ≈ encoder / z_led (paraxial approximation)
    sin_theta = led_pos / z_led
    # k in cycles/pixel: k_px = sin(θ) * dxp / λ
    # scaled to Nd: shift in pixel units of the Nd-pixel grid
    shift_px = sin_theta * dxp / wavelength * Nd
    return shift_px


def fpm_forward_single(
    obj_spectrum: np.ndarray,
    pupil: np.ndarray,
    shift_px: np.ndarray,
    Nd: int,
) -> np.ndarray:
    """
    Compute one FPM low-resolution image from the object spectrum.

    Given the high-resolution object spectrum Õ (No × No, DC at center):
    1. Extract the Nd × Nd sub-region at position (center + shift)
    2. Multiply by pupil
    3. IFFT → low-resolution image

    Parameters
    ----------
    obj_spectrum : ndarray, shape (No, No), complex
        High-resolution object spectrum (fftshift convention, DC at center).
    pupil : ndarray, shape (Nd, Nd), float
        Pupil mask (fftshift convention).
    shift_px : ndarray, shape (2,), float
        k-space shift in pixels (row_shift, col_shift).
    Nd : int
        Detector / low-resolution image size.

    Returns
    -------
    lr_image : ndarray, shape (Nd, Nd), float
        Simulated low-resolution intensity image.
    lr_field : ndarray, shape (Nd, Nd), complex
        Complex field at the detector (before taking |·|²).
    """
    No = obj_spectrum.shape[0]
    center = No // 2

    dr = int(round(shift_px[0]))
    dc = int(round(shift_px[1]))

    r0 = center + dr - Nd // 2
    c0 = center + dc - Nd // 2
    sub = obj_spectrum[r0: r0 + Nd, c0: c0 + Nd]

    # Apply pupil (in fftshift convention: DC at center)
    filtered = sub * pupil

    # IFFT → image plane
    lr_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(filtered)))
    lr_image = np.abs(lr_field) ** 2
    return lr_image, lr_field


def forward_model_stack(
    obj: np.ndarray,
    pupil: np.ndarray,
    encoder: np.ndarray,
    z_led: float,
    wavelength: float,
    dxp: float,
    Nd: int,
) -> np.ndarray:
    """
    Simulate the full FPM image stack for all LEDs.

    Parameters
    ----------
    obj : ndarray, shape (No, No), complex
        High-resolution complex object.
    pupil : ndarray, shape (Nd, Nd), float
        Pupil mask.
    encoder : ndarray, shape (J, 2), float
        LED positions [m].
    z_led, wavelength, dxp : float
        Physical parameters.
    Nd : int
        Detector size.

    Returns
    -------
    ptychogram : ndarray, shape (J, Nd, Nd), float
        Stack of simulated LR images.
    """
    No = obj.shape[0]
    obj_spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj)))
    J = len(encoder)
    ptychogram = np.zeros((J, Nd, Nd), dtype=np.float32)

    for j, led_pos in enumerate(encoder):
        shift = compute_kspace_shift(led_pos, z_led, wavelength, Nd, dxp)
        img, _ = fpm_forward_single(obj_spectrum, pupil, shift, Nd)
        ptychogram[j] = img.astype(np.float32)

    return ptychogram
