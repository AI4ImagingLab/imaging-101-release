"""
Forward model for conventional ptychography (CP).

In CP, the object is raster-scanned by a localized probe (illumination beam).
For each scan position j, the exit wave ψ_j is formed by the product of the
probe P and a small region (patch) of the object O:

    ψ_j(r) = P(r) · O(r - r_j)                                   (thin object)

The exit wave propagates to the detector. In the far field (Fraunhofer regime)
the detector intensity is:

    I_j(q) = |FT{ψ_j}(q)|²

For finite propagation distance z_o (near field), the Angular Spectrum
Propagator (ASP) replaces the simple FT:

    I_j(q) = |D_{r→q}[ψ_j](q)|²,   D = ASP or Fraunhofer

All propagation is handled by PtyLab's Operators module.
"""

import numpy as np
from .utils import aspw, fft2c


def get_object_patch(obj: np.ndarray, position: tuple, Np: int) -> np.ndarray:
    """
    Extract the object patch at scan position (row, col).

    Parameters
    ----------
    obj : ndarray, shape (No, No)
        Full complex object array.
    position : (int, int)
        (row, col) upper-left corner of the probe window.
    Np : int
        Probe / patch size in pixels.

    Returns
    -------
    patch : ndarray, shape (Np, Np), complex
        Object patch at this scan position.
    """
    row, col = position
    return obj[row: row + Np, col: col + Np].copy()


def compute_exit_wave(probe: np.ndarray, object_patch: np.ndarray) -> np.ndarray:
    """
    Compute the exit wave (thin-object approximation).

    The thin-element approximation (TEA) models the interaction between the
    incident probe and the object as a pointwise multiplication:

        ψ(r) = P(r) · O_j(r)

    Parameters
    ----------
    probe : ndarray, shape (..., Np, Np), complex
        Probe wavefield. May have leading batch dimensions.
    object_patch : ndarray, shape (Np, Np), complex
        Object patch at the current scan position.

    Returns
    -------
    esw : ndarray, shape (..., Np, Np), complex
        Exit surface wave.
    """
    return probe * object_patch


def fraunhofer_propagate(esw: np.ndarray) -> np.ndarray:
    """
    Propagate the exit wave to the detector plane (Fraunhofer / far-field).

    In the Fraunhofer approximation (valid when z_o >> Np² dxp² / λ), the
    detector field is proportional to the Fourier transform of the exit wave:

        Ψ(q) = FT{ψ}(q)

    where FT is the unitary centred DFT (fft2c).

    Parameters
    ----------
    esw : ndarray, shape (Np, Np) or (..., Np, Np), complex
        Exit surface wave.

    Returns
    -------
    detector_field : ndarray same shape, complex
        Complex field at the detector plane.
    """
    return fft2c(esw)


def asp_propagate(esw: np.ndarray, zo: float, wavelength: float, L: float) -> np.ndarray:
    """
    Propagate the exit wave to the detector plane via Angular Spectrum (ASP).

    The ASP handles both near- and far-field diffraction and is equivalent to
    Fraunhofer in the limit zo → ∞. It applies the transfer function:

        H(fx, fy) = exp(i 2π zo sqrt(1/λ² - fx² - fy²))

    Parameters
    ----------
    esw : ndarray, shape (Np, Np), complex
        Exit surface wave.
    zo : float
        Propagation distance [m].
    wavelength : float
        Illumination wavelength [m].
    L : float
        Physical size of the field of view [m] (= Np * dxp).

    Returns
    -------
    detector_field : ndarray, shape (Np, Np), complex
        Propagated field at the detector plane.
    """
    field, _ = aspw(esw, zo, wavelength, L)
    return field


def compute_detector_intensity(detector_field: np.ndarray) -> np.ndarray:
    """
    Compute the detector intensity from the propagated field.

    I(q) = |Ψ(q)|²

    Parameters
    ----------
    detector_field : ndarray, complex
        Complex field at the detector plane.

    Returns
    -------
    intensity : ndarray, float
        Intensity pattern (magnitude squared).
    """
    return np.abs(detector_field) ** 2


def forward_model(
    probe: np.ndarray,
    obj: np.ndarray,
    position: tuple,
    Np: int,
    propagator: str = "Fraunhofer",
    zo: float = None,
    wavelength: float = None,
    L: float = None,
) -> tuple:
    """
    Full CP forward model: object patch → diffraction intensity.

    Steps:
    1. Extract object patch at scan position
    2. Compute exit wave (TEA multiplication)
    3. Propagate to detector (Fraunhofer or ASP)
    4. Return intensity

    Parameters
    ----------
    probe : ndarray, shape (Np, Np), complex
        Probe wavefield.
    obj : ndarray, shape (No, No), complex
        Full object.
    position : (int, int)
        Scan position (row, col).
    Np : int
        Probe size in pixels.
    propagator : str
        'Fraunhofer' or 'ASP'.
    zo, wavelength, L : float
        Required only for 'ASP' propagator.

    Returns
    -------
    intensity : ndarray, shape (Np, Np), float
        Simulated detector intensity.
    esw : ndarray, shape (Np, Np), complex
        Exit surface wave (for diagnostics).
    """
    patch = get_object_patch(obj, position, Np)
    esw = compute_exit_wave(probe, patch)

    if propagator == "Fraunhofer":
        det_field = fraunhofer_propagate(esw)
    elif propagator == "ASP":
        det_field = asp_propagate(esw, zo, wavelength, L)
    else:
        raise ValueError(f"Unknown propagator '{propagator}'. Use 'Fraunhofer' or 'ASP'.")

    intensity = compute_detector_intensity(det_field)
    return intensity, esw
