"""Solvers for the Shack-Hartmann wavefront reconstruction task.

Single entry point: `reconstruct_all_levels` applies centroid estimation and
Tikhonov reconstruction to every WFE level, returning phase maps with per-level
NCC, NRMSE, and total reconstruction time.
"""

import time
import numpy as np

from .physics_model import (
    estimate_slopes,
    compute_reconstruction_matrix,
    reconstruct_wavefront,
    compute_ncc,
    compute_nrmse,
)


def reconstruct_all_levels(
    wfs_images: np.ndarray,
    ref_image: np.ndarray,
    detector_coords_x: np.ndarray,
    detector_coords_y: np.ndarray,
    subap_map: np.ndarray,
    response_matrix: np.ndarray,
    dm_modes: np.ndarray,
    aperture: np.ndarray,
    wavelength: float,
    n_valid_subaps: int,
    rcond: float = 1e-3,
    ground_truth_phases: np.ndarray = None,
) -> dict:
    """Reconstruct wavefront phase for every WFE level.

    Pipeline per level:
      1. Centroid estimation: wfs_image + ref_image → slopes  (via estimate_slopes)
      2. Tikhonov reconstruction: slopes → mode amplitudes → phase map

    Timing covers only steps 1–2 for all levels (not the one-time SVD in
    compute_reconstruction_matrix, which is a pre-computation step).

    Parameters
    ----------
    wfs_images          : (N_levels, N_det) or (N_levels, H, W)
    ref_image           : (N_det,) or (H, W)
    detector_coords_x   : (N_det,)  [m]
    detector_coords_y   : (N_det,)  [m]
    subap_map           : (N_det,)  int32
    response_matrix     : (N_slopes, N_modes)
    dm_modes            : (N_modes, N_pupil_px)
    aperture            : (N_pupil_px,)
    wavelength          : float  [m]
    n_valid_subaps      : int
    rcond               : float  – Tikhonov threshold
    ground_truth_phases : (N_levels, N_pupil_px) or None

    Returns
    -------
    dict with keys:
        reconstructed_phases    (N_levels, N_pupil_px)  float32
        ncc_per_level           (N_levels,)  float64  (only if GT given)
        nrmse_per_level         (N_levels,)  float64  (only if GT given)
        reconstruction_time_s   float  — wall-clock seconds for all levels
                                         (centroid extraction + reconstruction only)
    """
    M    = compute_reconstruction_matrix(
        response_matrix.astype(np.float64), rcond)
    mask = aperture > 0.5
    N_levels   = wfs_images.shape[0]
    N_pupil_px = dm_modes.shape[1]

    phases_out = np.zeros((N_levels, N_pupil_px))
    ncc_arr    = np.zeros(N_levels)
    nrmse_arr  = np.zeros(N_levels)

    ref_f64 = ref_image.astype(np.float64)
    cx_f64  = detector_coords_x.astype(np.float64)
    cy_f64  = detector_coords_y.astype(np.float64)

    t0 = time.perf_counter()

    for i in range(N_levels):
        # Step 1: centroid estimation → slope differences
        slopes = estimate_slopes(
            wfs_images[i].astype(np.float64),
            ref_f64, cx_f64, cy_f64, subap_map, n_valid_subaps,
        )

        # Step 2: Tikhonov reconstruction → phase map
        phi = reconstruct_wavefront(
            slopes.astype(np.float64), M,
            dm_modes.astype(np.float64), wavelength,
        )
        phases_out[i] = phi

        if ground_truth_phases is not None:
            gt = ground_truth_phases[i].astype(np.float64)
            ncc_arr[i]   = compute_ncc(phi, gt, mask)
            nrmse_arr[i] = compute_nrmse(phi, gt, mask)

    reconstruction_time_s = time.perf_counter() - t0

    result = {
        'reconstructed_phases':  phases_out.astype(np.float32),
        'reconstruction_time_s': reconstruction_time_s,
    }
    if ground_truth_phases is not None:
        result['ncc_per_level']   = ncc_arr
        result['nrmse_per_level'] = nrmse_arr
    return result
