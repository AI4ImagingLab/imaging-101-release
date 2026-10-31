"""Physics-based forward models for the Shack-Hartmann wavefront sensing task.

The Shack-Hartmann WFS measures local wavefront gradients. The full pipeline
maps raw detector images to wavefront phase:

  1. Centroid estimation: raw image → WFS slopes
       slopes = estimate_slopes(wfs_image, ref_image, coords_x, coords_y, subap_map, N_valid)

  2. Mode reconstruction: slopes → DM mode amplitudes
       a_hat = M @ (slopes)    where M = R⁺ (Tikhonov pseudo-inverse)

  3. Phase assembly: mode amplitudes → wavefront phase
       phi = 4π/λ × (dm_modes.T @ a_hat)

The forward model maps DM mode coefficients to WFS slopes:
    s = R @ a

where R (N_slopes × N_modes) is the calibrated response matrix.
"""

import numpy as np


def estimate_slopes(
    wfs_image: np.ndarray,
    ref_image: np.ndarray,
    detector_coords_x: np.ndarray,
    detector_coords_y: np.ndarray,
    subap_map: np.ndarray,
    n_valid_subaps: int,
) -> np.ndarray:
    """Weighted-centroid Shack-Hartmann slope estimator.

    For each valid subaperture, computes the intensity-weighted centroid of the
    WFS spot and of the reference spot, then returns the displacement (slope):

        Δcx[j] = Σ(x_i * I_i) / Σ(I_i)  −  Σ(x_i * I_ref_i) / Σ(I_ref_i)

    The output ordering is [Δcx_0, …, Δcx_{N-1}, Δcy_0, …, Δcy_{N-1}]
    (all x-slopes first, then y-slopes), matching HCIPy's
    ShackHartmannWavefrontSensorEstimator.estimate() convention.

    Parameters
    ----------
    wfs_image          : (N_det,) or (H, W)  raw SH-WFS detector image [photons]
    ref_image          : (N_det,) or (H, W)  reference (flat wavefront) image [photons]
    detector_coords_x  : (N_det,) or (H, W)  focal-plane x coordinate per pixel [m]
    detector_coords_y  : (N_det,) or (H, W)  focal-plane y coordinate per pixel [m]
    subap_map          : (N_det,) or (H, W)  int32 — subaperture rank per pixel
                                   (-1 = not part of any valid subaperture,
                                    0..n_valid_subaps-1 = subaperture index)
    n_valid_subaps     : int        number of valid subapertures

    Returns
    -------
    slopes : (2 * n_valid_subaps,)  [all Δcx | all Δcy] in focal-plane coords [m]
    """
    slopes_x = np.zeros(n_valid_subaps)
    slopes_y = np.zeros(n_valid_subaps)

    # Accept either 1D (N_det,) or 2D (H, W) inputs
    wfs_image         = np.asarray(wfs_image).ravel()
    ref_image         = np.asarray(ref_image).ravel()
    detector_coords_x = np.asarray(detector_coords_x).ravel()
    detector_coords_y = np.asarray(detector_coords_y).ravel()
    subap_map         = np.asarray(subap_map).ravel()

    for j in range(n_valid_subaps):
        mask   = (subap_map == j)
        I      = wfs_image[mask]
        I_ref  = ref_image[mask]
        x      = detector_coords_x[mask]
        y      = detector_coords_y[mask]

        total     = I.sum()
        total_ref = I_ref.sum()

        if total > 0 and total_ref > 0:
            cx     = (x * I).sum()     / total
            cy     = (y * I).sum()     / total
            cx_ref = (x * I_ref).sum() / total_ref
            cy_ref = (y * I_ref).sum() / total_ref
        else:
            cx = cy = cx_ref = cy_ref = 0.0

        slopes_x[j] = cx - cx_ref
        slopes_y[j] = cy - cy_ref

    return np.concatenate([slopes_x, slopes_y])


def compute_reconstruction_matrix(
    response_matrix: np.ndarray,
    rcond: float = 1e-3,
) -> np.ndarray:
    """Tikhonov pseudo-inverse of the WFS response matrix.

    R maps DM actuator amplitudes to WFS slope measurements:
        s = R @ a      (N_slopes,) = (N_slopes, N_modes) @ (N_modes,)

    The reconstruction matrix M = R⁺ inverts this:
        a_recon = M @ slopes

    Regularisation: singular modes with σᵢ < rcond × σ_max are zeroed.

    Parameters
    ----------
    response_matrix : (N_slopes, N_modes)
    rcond           : float  – regularisation threshold

    Returns
    -------
    reconstruction_matrix : (N_modes, N_slopes)
    """
    U, s, Vt = np.linalg.svd(response_matrix, full_matrices=False)
    threshold = rcond * s.max()
    s_inv     = np.where(s > threshold, 1.0 / s, 0.0)
    return (Vt.T * s_inv) @ U.T


def reconstruct_wavefront(
    slopes: np.ndarray,
    reconstruction_matrix: np.ndarray,
    dm_modes: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    """Reconstruct the wavefront phase from WFS slope differences.

    Steps:
      1. Reconstruct DM mode amplitudes: a = M @ slopes
      2. Convert to phase: φ = 4π × (dm_modes.T @ a) / λ

    The DM is modelled as a double-pass reflection element, so the phase
    contribution per unit surface height is 4π/λ.

    Parameters
    ----------
    slopes               : (N_slopes,)      slope differences (s − s_ref)
    reconstruction_matrix: (N_modes, N_slopes)
    dm_modes             : (N_modes, N_pupil_px)
    wavelength           : float  [m]

    Returns
    -------
    wavefront_phase : (N_pupil_px,)  [rad]
    """
    a_recon = reconstruction_matrix @ slopes          # (N_modes,)
    surface = dm_modes.T @ a_recon                    # (N_pupil_px,)
    return 4.0 * np.pi * surface / wavelength         # [rad]


def compute_ncc(estimate: np.ndarray, reference: np.ndarray,
                mask: np.ndarray = None) -> float:
    """Normalised cross-correlation (cosine similarity) over the aperture.

    NCC = (x̂ · x_ref) / (‖x̂‖ ‖x_ref‖)

    Parameters
    ----------
    estimate  : (N_pupil_px,)
    reference : (N_pupil_px,)
    mask      : (N_pupil_px,) bool, optional

    Returns
    -------
    float in [−1, 1]
    """
    if mask is not None:
        estimate  = estimate[mask]
        reference = reference[mask]
    norm = np.linalg.norm(estimate) * np.linalg.norm(reference)
    if norm == 0:
        return 0.0
    return float(np.dot(estimate, reference) / norm)


def compute_nrmse(estimate: np.ndarray, reference: np.ndarray,
                  mask: np.ndarray = None) -> float:
    """Normalised RMSE relative to the dynamic range of the reference.

    NRMSE = RMS(x̂ − x_ref) / (max(x_ref) − min(x_ref))

    Parameters
    ----------
    estimate  : (N_pupil_px,)
    reference : (N_pupil_px,)
    mask      : (N_pupil_px,) bool, optional

    Returns
    -------
    float ≥ 0
    """
    if mask is not None:
        estimate  = estimate[mask]
        reference = reference[mask]
    rms       = float(np.sqrt(np.mean((estimate - reference) ** 2)))
    dyn_range = float(reference.max() - reference.min())
    if dyn_range == 0:
        return 0.0
    return rms / dyn_range
