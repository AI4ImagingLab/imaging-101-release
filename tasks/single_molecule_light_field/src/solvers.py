"""3D fitting solvers for Single-Molecule Light Field Microscopy.

Implements the multi-view parallax inversion: given 2D localisations
observed in multiple microlens views, estimates 3D positions (x0, y0, z)
via ordinary least squares (OLS).

Extracted from: hexSMLFM / PySMLFM (TheLeeLab / Photometrics, Cambridge).
Reference: R. R. Sims et al., Optica 7, 1065 (2020).

The inverse problem
-------------------
For molecule k observed in view i:
    x_i = x0 + u_i/rho + z*alpha_u_i
    y_i = y0 + v_i/rho + z*alpha_v_i

Rearranging:
    b_x_i = x_i - u_i/rho = x0 + z*alpha_u_i
    b_y_i = y_i - v_i/rho = y0 + z*alpha_v_i

Stacking all views gives A * [x0, y0, z]^T = b, solved by np.linalg.lstsq.

Pipeline
--------
1. fit_aberrations : first-pass fitting on ≤1000 frames → per-view (dx,dy)
2. fit_3d_localizations : full-dataset fitting on corrected localisations
"""

import dataclasses
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .physics_model import FourierMicroscope, Localisations


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------

@dataclass
class FitParams:
    """Configuration for a single fitting pass.

    Attributes:
        frame_min:       First frame to process (-1 → use data minimum).
        frame_max:       Last frame to process  (-1 → use data maximum or 1000).
        disparity_max:   Search range for Z in disparity units (microns).
        disparity_step:  Step size for the disparity grid (microns).
        dist_search:     Acceptance window around the best-Z disparity (microns).
        angle_tolerance: Max angle between (Δx,Δy) and (Δu,Δv) vectors (degrees).
        threshold:       Max OLS residual to accept a fit (microns).
        min_views:       Minimum number of views required for a valid fit.
        z_calib:         Optional calibration factor: physical Z = optical Z * z_calib.
    """
    frame_min: int
    frame_max: int
    disparity_max: float
    disparity_step: float
    dist_search: float
    angle_tolerance: float
    threshold: float
    min_views: int
    z_calib: Optional[float] = None


@dataclass
class AberrationParams:
    """Criteria for selecting molecules to use in aberration estimation.

    Attributes:
        axial_window:     Keep only molecules with |z| < axial_window (microns).
        photon_threshold: Keep only molecules with total photons > threshold.
        min_views:        Keep only fits with at least this many views.
    """
    axial_window: float
    photon_threshold: int
    min_views: int


@dataclass
class FitData:
    """Per-molecule fitting result (used internally for aberration correction)."""
    frame: int
    model: npt.NDArray[float]   # [x0, y0, z]
    points: npt.NDArray[float]  # (K, 13) contributing localisations
    photon_count: float
    std_err: npt.NDArray[float] # [se_x, se_y, se_z]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_aberrations(
    lfl: Localisations,
    lfm: FourierMicroscope,
    meta: dict,
    worker_count: int = 0,
) -> npt.NDArray[float]:
    """First-pass fitting on up to 1000 frames to estimate per-view aberrations.

    Each microlens view may have a small systematic XY offset from optical
    aberrations.  This function fits a subset of molecules near focus,
    computes the mean residual in each view, and returns a correction table.

    Call lfl.correct_xy(correction) afterwards before the full fitting pass.

    Args:
        lfl:          Localisations with alpha_uv populated (filtered_locs_2d).
        lfm:          FourierMicroscope instance.
        meta:         meta_data.json dict (uses fit_params_aberration and
                      aberration_params keys).
        worker_count: Ignored (kept for API compatibility; fitting is serial).

    Returns:
        correction (V, 5): one row per unique view (u, v):
            [0] u  [1] v  [2] dx_correction µm  [3] dy_correction µm  [4] n_pts
    """
    p = meta["fit_params_aberration"]
    fit_params = FitParams(
        frame_min=p["frame_min"] if p["frame_min"] > 0 else lfl.min_frame,
        frame_max=(p["frame_max"] if p["frame_max"] > 0
                   else min(1000, lfl.max_frame)),
        disparity_max=p["disparity_max"],
        disparity_step=p["disparity_step"],
        dist_search=p["dist_search"],
        angle_tolerance=p["angle_tolerance"],
        threshold=p["threshold"],
        min_views=p["min_views"],
        z_calib=None,
    )

    ab = meta["aberration_params"]
    aberration_params = AberrationParams(
        axial_window=ab["axial_window"],
        photon_threshold=ab["photon_threshold"],
        min_views=ab["min_views"],
    )

    _, fit_data = _light_field_fit(lfl.filtered_locs_2d, lfm.rho_scaling, fit_params)
    correction  = _calculate_view_error(lfl.filtered_locs_2d, lfm.rho_scaling,
                                        fit_data, aberration_params)
    return correction


def fit_3d_localizations(
    lfl: Localisations,
    lfm: FourierMicroscope,
    meta: dict,
    worker_count: int = 0,
    progress_func: Optional[Callable[[int, int, int], None]] = None,
) -> npt.NDArray[float]:
    """Full 3D fitting on aberration-corrected localisation data.

    Args:
        lfl:           Localisations with corrected_locs_2d populated.
        lfm:           FourierMicroscope instance.
        meta:          meta_data.json dict (uses fit_params_full key).
        worker_count:  Ignored (kept for API compatibility).
        progress_func: Optional callable(frame, min_frame, max_frame).

    Returns:
        locs_3d (M, 8): one row per reconstructed 3D localisation:
            [0] X µm  [1] Y µm  [2] Z µm (z_calib applied)
            [3] lateral fit error µm  [4] axial fit error µm
            [5] n_views  [6] photons  [7] frame
    """
    locs_in = (lfl.corrected_locs_2d
               if lfl.corrected_locs_2d is not None
               else lfl.filtered_locs_2d)

    p = meta["fit_params_full"]
    fit_params = FitParams(
        frame_min=p["frame_min"] if p["frame_min"] > 0 else lfl.min_frame,
        frame_max=p["frame_max"] if p["frame_max"] > 0 else lfl.max_frame,
        disparity_max=p["disparity_max"],
        disparity_step=p["disparity_step"],
        dist_search=p["dist_search"],
        angle_tolerance=p["angle_tolerance"],
        threshold=p["threshold"],
        min_views=p["min_views"],
        z_calib=p.get("z_calib", None),
    )

    locs_3d, _ = _light_field_fit(
        locs_in, lfm.rho_scaling, fit_params,
        progress_func=progress_func,
    )
    return locs_3d


# ---------------------------------------------------------------------------
# Core fitting algorithm
# ---------------------------------------------------------------------------

def _light_field_fit(
    locs_2d: npt.NDArray[float],
    rho_scaling: float,
    fit_params: FitParams,
    progress_func: Optional[Callable[[int, int, int], None]] = None,
    progress_step: int = 1000,
) -> Tuple[npt.NDArray[float], List[FitData]]:
    """Frame-by-frame 3D fitting loop.

    For each frame, repeatedly selects a seed localisation, groups companion
    localisations from other views by parallax consistency, fits the OLS model,
    and accepts if the residual is below the threshold.

    Returns:
        fitted_points (M, 8), total_fit list of FitData objects.
    """
    min_frame = fit_params.frame_min
    max_frame = fit_params.frame_max
    min_views = fit_params.min_views
    threshold = fit_params.threshold
    z_calib   = fit_params.z_calib

    fitted_points = np.empty((0, 8))
    total_fit: List[FitData] = []

    progress_next = min_frame + progress_step

    for frame in range(min_frame, max_frame + 1):

        if progress_func is not None:
            if frame >= progress_next or frame == max_frame:
                progress_func(frame, min_frame, max_frame)
                progress_next += progress_step

        # Candidates for this frame, central views first then by intensity
        candidates = locs_2d[locs_2d[:, 0] == frame].copy()
        if candidates.shape[0] == 0:
            continue

        u, v = candidates[:, 1], candidates[:, 2]
        cen_sel = (np.abs(u) < 0.1) & (np.abs(v) < 0.1)
        loc_cen = candidates[cen_sel][np.argsort(-candidates[cen_sel, 7])]
        loc_out = candidates[~(u == 0) | ~(v == 0)]
        loc_out = loc_out[np.argsort(-loc_out[:, 7])]
        candidates = np.row_stack((loc_cen, loc_out)) if loc_out.size else loc_cen

        reps = 0
        while candidates.shape[0] > min_views and reps < 100:
            reps += 1
            seed = candidates[0]
            loc_fit, view_count, _ = _group_localisations(
                seed, candidates, fit_params, rho_scaling)

            if view_count < min_views:
                candidates = candidates[1:]
                continue

            model, std_err, _ = _get_backward_model(loc_fit, rho_scaling)
            dist = np.sqrt(np.sum(std_err ** 2))
            if dist > threshold:
                candidates = candidates[1:]
                continue

            # Remove fitted localisations from pool
            used = np.all(candidates[:, None] == loc_fit, axis=-1).any(-1)
            candidates = candidates[~used]

            if loc_fit.size > 0:
                photon_count = np.sum(loc_fit[:, 7])
                fitted_points = np.row_stack((fitted_points, [
                    model[0], model[1], model[2],
                    np.mean(std_err[0:2]), std_err[2],
                    view_count + 1, photon_count, frame,
                ]))
                total_fit.append(FitData(
                    frame=frame, model=model, points=loc_fit.copy(),
                    photon_count=photon_count, std_err=std_err,
                ))

    if z_calib is not None and fitted_points.shape[0] > 0:
        fitted_points[:, 2] *= z_calib

    return fitted_points, total_fit


def _group_localisations(
    seed: npt.NDArray[float],
    in_points: npt.NDArray[float],
    fit_params: FitParams,
    rho_scaling: float,
) -> Tuple[npt.NDArray[float], int, int]:
    """Group companion localisations consistent with seed's parallax.

    Applies two selection criteria:
    1. Angle test: the direction of (Δx, Δy) must agree with (Δu, Δv)
       within angle_tolerance.
    2. Disparity test: (dxy − duv/rho) / duv ≈ z must cluster at one Z.

    Returns:
        candidates (K, 13), view_count (int), duplicate_count (int).
    """
    angle_tol    = np.deg2rad(fit_params.angle_tolerance)
    max_disparity = fit_params.disparity_max
    dis_tol      = fit_params.dist_search
    dz           = fit_params.disparity_step

    su, sv = seed[1], seed[2]
    sx, sy = seed[3], seed[4]

    points = in_points.copy()
    # Remove same-view localisations
    points = points[~((points[:, 1] == su) & (points[:, 2] == sv))]

    du = points[:, 1] - su
    dv = points[:, 2] - sv
    dx = points[:, 3] - sx
    dy = points[:, 4] - sy

    # 1. Angle test
    angles_uv = np.arctan2(dv, du)
    angles_xy = np.arctan2(dy, dx)
    diff = np.abs(angles_xy - angles_uv)
    angle_sel = (diff < angle_tol) | (diff > (np.pi - angle_tol))
    points = points[angle_sel]

    # 2. Disparity test
    du = points[:, 1] - su
    dv = points[:, 2] - sv
    dx = points[:, 3] - sx
    dy = points[:, 4] - sy
    dxy = np.sqrt(dx ** 2 + dy ** 2)
    duv = np.sqrt(du ** 2 + dv ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        disparity = np.where(duv > 0, (dxy - duv / rho_scaling) / duv, 0.0)

    range_z   = np.arange(-max_disparity, max_disparity + dz / 2, dz)
    num_pts   = np.array([
        np.sum((disparity > z_i - dis_tol) & (disparity <= z_i + dis_tol))
        for z_i in range_z
    ])
    best_z    = range_z[np.argmax(num_pts)]
    z_sel     = (disparity <= best_z - dis_tol) | (disparity > best_z + dis_tol)
    points    = points[~z_sel]

    # Resolve duplicate views (multiple localisations in same lens)
    candidates    = seed
    view_count    = 0
    duplicate_count = 0

    if points.size > 0:
        uv_vals, uv_inv, uv_cnt = np.unique(
            points[:, 1:3], axis=0, return_inverse=True, return_counts=True)

        indices = [np.where(np.all(points[:, 1:3] == uv, axis=1))[0]
                   for uv in uv_vals]

        candidates = np.row_stack(
            (candidates, points[np.isin(uv_inv, np.nonzero(uv_cnt == 1))]))

        dup_indices = [idx for idx in indices if len(idx) > 1]
        view_count  = len(uv_vals)
        duplicate_count = len(dup_indices)

        for view_ind in dup_indices:
            d = np.array([
                np.sqrt(np.sum(
                    _get_backward_model(
                        np.row_stack((candidates, points[j])), rho_scaling
                    )[1] ** 2
                ))
                for j in view_ind
            ])
            candidates = np.row_stack((candidates, points[view_ind[np.argmin(d)]]))

        # Reject if two views share identical U or V (degenerate)
        if view_count == 1:
            if np.var(candidates[:, 1], ddof=1) < 0.1 or \
               np.var(candidates[:, 2], ddof=1) < 0.1:
                candidates = np.array([])
                view_count = 0

    return candidates, view_count, duplicate_count


def _get_backward_model(
    locs: npt.NDArray[float],
    rho_scaling: float,
) -> Tuple[npt.NDArray[float], npt.NDArray[float], float]:
    """Solve OLS for (x0, y0, z) from multi-view localisations.

    System:  A * [x0, y0, z]^T = b
    where for each view i:
        A rows: [1, 0, alpha_u_i] and [0, 1, alpha_v_i]
        b vals: x_i - u_i/rho    and  y_i - v_i/rho

    Returns:
        model (3,): [x0, y0, z]
        std_err (3,): standard errors [se_x0, se_y0, se_z]
        mse (float): mean squared error of the residuals
    """
    u       = locs[:, 1]
    v       = locs[:, 2]
    x       = locs[:, 3] - u / rho_scaling
    y       = locs[:, 4] - v / rho_scaling
    alpha_u = locs[:, 10]
    alpha_v = locs[:, 11]

    ones  = np.ones_like(alpha_u)
    zeros = np.zeros_like(alpha_u)

    A = np.row_stack((
        np.column_stack((ones, zeros, alpha_u)),
        np.column_stack((zeros, ones, alpha_v)),
    ))
    b = np.concatenate((x, y))

    model, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    mse = (np.sum(residuals) / (A.shape[0] - rank)
           if residuals.size > 0 else 0.0)
    cov     = mse * np.linalg.inv(A.T @ A)
    std_err = np.sqrt(np.diag(cov))

    return model, std_err, mse


def _get_forward_model_error(
    model: npt.NDArray[float],
    points: npt.NDArray[float],
    rho_scaling: float,
) -> npt.NDArray[float]:
    """Compute per-view residuals from a fitted model.

    residual_i = (x_i, y_i) - (u_i/rho + x0 + z*alpha_i)

    Returns:
        (K, 2) array of (dx, dy) residuals.
    """
    uv       = points[:, 1:3]
    xy       = points[:, 3:5]
    alpha_uv = points[:, 10:12]
    xy0      = model[0:2]
    z        = model[2]
    return xy - (uv / rho_scaling) - xy0 - (z * alpha_uv)


def _calculate_view_error(
    locs_2d: npt.NDArray[float],
    rho_scaling: float,
    fit_data: List[FitData],
    aberration_params: AberrationParams,
) -> npt.NDArray[float]:
    """Compute the mean per-view fit residual across all accepted molecules.

    Used to estimate systematic per-view offsets (aberrations).

    Args:
        locs_2d:           Filtered localisation array (N, 13).
        rho_scaling:       ρ scaling factor.
        fit_data:          List of FitData from _light_field_fit.
        aberration_params: Quality thresholds for selecting molecules.

    Returns:
        correction (V, 5): one row per unique view:
            [0] u  [1] v  [2] mean_dx  [3] mean_dy  [4] n_points_used
    """
    views      = np.unique(locs_2d[:, 1:3], axis=0)
    view_count = views.shape[0]
    correction = np.zeros((view_count, 5))
    correction[:, 0:2] = views

    min_views = aberration_params.min_views
    ph_thresh = aberration_params.photon_threshold
    ax_win    = aberration_params.axial_window

    for fd in fit_data:
        if (fd.points.shape[0] <= min_views
                or abs(fd.model[2]) >= ax_win
                or fd.photon_count <= ph_thresh):
            continue
        fit_err = _get_forward_model_error(fd.model, fd.points, rho_scaling)
        for j in range(view_count):
            mask = ((fd.points[:, 1] == views[j, 0])
                    & (fd.points[:, 2] == views[j, 1]))
            if mask.any():
                correction[j, 2:4] += fit_err[mask][0]
                correction[j, 4]   += 1

    # Average: avoid division by zero
    c = correction[:, 4]
    correction[:, 2] = np.where(c > 0, correction[:, 2] / c, 0.0)
    correction[:, 3] = np.where(c > 0, correction[:, 3] / c, 0.0)

    return correction
