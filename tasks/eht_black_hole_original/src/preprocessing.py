"""
Preprocessing: Load Observations and Compute Closure Quantities
================================================================

Loads raw VLBI data (visibilities, uv-coordinates, station info) and
computes closure phases and log closure amplitudes with noise estimates.

Functions
---------
load_observation   Load raw_data.npz
load_metadata      Load meta_data JSON
find_triangles     Enumerate independent closure-phase triangles
find_quadrangles   Enumerate independent closure-amplitude quadrangles
compute_closure_phases           Complex vis → closure phases (radians)
compute_log_closure_amplitudes   Complex vis → log closure amplitudes
closure_phase_sigma              Error propagation for closure phase noise
closure_amplitude_sigma          Error propagation for log closure amp noise
prepare_data       Combined loader returning all data structures
"""

import os
import json
import numpy as np
from itertools import combinations


def load_observation(data_dir: str = "data") -> dict:
    """
    Load raw observation data from raw_data.npz.

    Returns
    -------
    dict with keys:
        vis_cal       : (M,) complex — calibrated visibilities
        vis_corrupt   : (M,) complex — gain-corrupted visibilities
        uv_coords     : (M, 2) float — baseline (u,v) in wavelengths
        sigma_vis     : (M,) float — per-baseline noise σ (Jy)
        station_ids   : (M, 2) int — station pair indices
        And closure data from ehtim (per-scan):
        cp_values_deg, cp_sigmas_deg, cp_u1/u2/u3 — calibrated closure phases
        lca_values, lca_sigmas, lca_u1/u2/u3/u4 — calibrated log closure amps
        cp_corrupt_values_deg, etc. — corrupted closure quantities
    """
    path = os.path.join(data_dir, "raw_data.npz")
    d = np.load(path)
    result = {}
    for key in d.files:
        result[key] = d[key]
    return result


def load_metadata(data_dir: str = "data") -> dict:
    """
    Load imaging parameters from meta_data (JSON).

    Returns
    -------
    dict with imaging parameters (N, pixel_size_rad, etc.)
    """
    path = os.path.join(data_dir, "meta_data.json")
    with open(path) as f:
        return json.load(f)


def find_triangles(station_ids: np.ndarray, n_stations: int) -> np.ndarray:
    """
    Enumerate independent closure-phase triangles.

    For each scan (all baselines share the same set of stations),
    find a minimal set of triangles. A minimal set has N_s - 1 choose 2
    triangles for N_s stations (one per independent closure phase).

    Uses the minimum-spanning-tree approach: pick triangles from all
    combinations of 3 stations that form valid baselines.

    Parameters
    ----------
    station_ids : (M, 2) int — station pair indices
    n_stations  : int — number of stations

    Returns
    -------
    triangles : (N_tri, 3) int — station index triples (i, j, k)
    """
    # Build set of available baselines
    baselines = set()
    for row in station_ids:
        i, j = int(row[0]), int(row[1])
        baselines.add((min(i, j), max(i, j)))

    triangles = []
    for combo in combinations(range(n_stations), 3):
        i, j, k = combo
        if ((i, j) in baselines and
            (j, k) in baselines and
            (i, k) in baselines):
            triangles.append([i, j, k])

    return np.array(triangles, dtype=np.int64)


def find_quadrangles(station_ids: np.ndarray, n_stations: int) -> np.ndarray:
    """
    Enumerate independent closure-amplitude quadrangles.

    A quadrangle (i, j, k, l) gives log closure amplitude:
        log|V_ij| + log|V_kl| - log|V_ik| - log|V_jl|

    Parameters
    ----------
    station_ids : (M, 2) int
    n_stations  : int

    Returns
    -------
    quadrangles : (N_quad, 4) int — station index quads (i, j, k, l)
    """
    baselines = set()
    for row in station_ids:
        i, j = int(row[0]), int(row[1])
        baselines.add((min(i, j), max(i, j)))

    quadrangles = []
    for combo in combinations(range(n_stations), 4):
        i, j, k, l = combo
        # Need all 6 baselines for the 4 stations
        pairs = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
        if all(p in baselines for p in pairs):
            quadrangles.append([i, j, k, l])

    return np.array(quadrangles, dtype=np.int64)


def _find_baseline(station_ids, s1, s2):
    """Find baseline index for station pair (s1, s2) or (s2, s1).

    Returns (index, conjugate) where conjugate is True if the baseline
    is stored as (s2, s1) and the visibility must be conjugated.
    """
    # Forward match
    mask_fwd = (station_ids[:, 0] == s1) & (station_ids[:, 1] == s2)
    idx_fwd = np.where(mask_fwd)[0]
    if len(idx_fwd) > 0:
        return int(idx_fwd[0]), False

    # Reverse match
    mask_rev = (station_ids[:, 0] == s2) & (station_ids[:, 1] == s1)
    idx_rev = np.where(mask_rev)[0]
    if len(idx_rev) > 0:
        return int(idx_rev[0]), True

    raise ValueError(f"Baseline ({s1}, {s2}) not found")


def compute_closure_phases(
    vis: np.ndarray,
    station_ids: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """
    Compute closure phases from complex visibilities.

    Closure phase for triangle (i, j, k):
        φ_C = arg(V_ij * V_jk * V_ki)

    Parameters
    ----------
    vis        : (M,) complex visibilities
    station_ids: (M, 2) int station pairs
    triangles  : (N_tri, 3) int station triples

    Returns
    -------
    cphase : (N_tri,) float — closure phases in radians
    """
    n_tri = len(triangles)
    cphase = np.zeros(n_tri)

    for t in range(n_tri):
        i, j, k = triangles[t]

        idx_ij, conj_ij = _find_baseline(station_ids, i, j)
        idx_jk, conj_jk = _find_baseline(station_ids, j, k)
        idx_ki, conj_ki = _find_baseline(station_ids, k, i)

        v_ij = np.conj(vis[idx_ij]) if conj_ij else vis[idx_ij]
        v_jk = np.conj(vis[idx_jk]) if conj_jk else vis[idx_jk]
        v_ki = np.conj(vis[idx_ki]) if conj_ki else vis[idx_ki]

        cphase[t] = np.angle(v_ij * v_jk * v_ki)

    return cphase


def compute_log_closure_amplitudes(
    vis: np.ndarray,
    station_ids: np.ndarray,
    quadrangles: np.ndarray,
) -> np.ndarray:
    """
    Compute log closure amplitudes from complex visibilities.

    For quadrangle (i, j, k, l):
        log CA = log|V_ij| + log|V_kl| - log|V_ik| - log|V_jl|

    Parameters
    ----------
    vis         : (M,) complex visibilities
    station_ids : (M, 2) int
    quadrangles : (N_quad, 4) int station quads

    Returns
    -------
    logcamp : (N_quad,) float
    """
    n_quad = len(quadrangles)
    logcamp = np.zeros(n_quad)

    for q in range(n_quad):
        i, j, k, l = quadrangles[q]

        idx_ij, _ = _find_baseline(station_ids, i, j)
        idx_kl, _ = _find_baseline(station_ids, k, l)
        idx_ik, _ = _find_baseline(station_ids, i, k)
        idx_jl, _ = _find_baseline(station_ids, j, l)

        logcamp[q] = (np.log(np.abs(vis[idx_ij])) + np.log(np.abs(vis[idx_kl]))
                     - np.log(np.abs(vis[idx_ik])) - np.log(np.abs(vis[idx_jl])))

    return logcamp


def closure_phase_sigma(
    sigma_vis: np.ndarray,
    vis: np.ndarray,
    station_ids: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """
    Noise propagation for closure phases.

    σ_CP = sqrt(σ₁²/|V₁|² + σ₂²/|V₂|² + σ₃²/|V₃|²)

    Returns closure phase sigma in radians.
    """
    n_tri = len(triangles)
    sigma_cp = np.zeros(n_tri)

    for t in range(n_tri):
        i, j, k = triangles[t]
        idx_ij, _ = _find_baseline(station_ids, i, j)
        idx_jk, _ = _find_baseline(station_ids, j, k)
        idx_ki, _ = _find_baseline(station_ids, k, i)

        var_sum = (sigma_vis[idx_ij]**2 / np.abs(vis[idx_ij])**2 +
                   sigma_vis[idx_jk]**2 / np.abs(vis[idx_jk])**2 +
                   sigma_vis[idx_ki]**2 / np.abs(vis[idx_ki])**2)
        sigma_cp[t] = np.sqrt(var_sum)

    return sigma_cp


def closure_amplitude_sigma(
    sigma_vis: np.ndarray,
    vis: np.ndarray,
    station_ids: np.ndarray,
    quadrangles: np.ndarray,
) -> np.ndarray:
    """
    Noise propagation for log closure amplitudes.

    σ_logCA = sqrt(1/SNR₁² + 1/SNR₂² + 1/SNR₃² + 1/SNR₄²)

    where SNR_k = |V_k| / σ_k.
    """
    n_quad = len(quadrangles)
    sigma_lca = np.zeros(n_quad)

    for q in range(n_quad):
        i, j, k, l = quadrangles[q]
        idx_ij, _ = _find_baseline(station_ids, i, j)
        idx_kl, _ = _find_baseline(station_ids, k, l)
        idx_ik, _ = _find_baseline(station_ids, i, k)
        idx_jl, _ = _find_baseline(station_ids, j, l)

        var_sum = (sigma_vis[idx_ij]**2 / np.abs(vis[idx_ij])**2 +
                   sigma_vis[idx_kl]**2 / np.abs(vis[idx_kl])**2 +
                   sigma_vis[idx_ik]**2 / np.abs(vis[idx_ik])**2 +
                   sigma_vis[idx_jl]**2 / np.abs(vis[idx_jl])**2)
        sigma_lca[q] = np.sqrt(var_sum)

    return sigma_lca


def prepare_data(data_dir: str = "data") -> tuple:
    """
    Load all data and compute closure quantities.

    Returns
    -------
    obs : dict — observation data
    closure : dict — closure phases, log closure amps, sigmas, triangles, quadrangles
    meta : dict — imaging parameters
    """
    obs = load_observation(data_dir)
    meta = load_metadata(data_dir)

    n_stations = meta["n_stations"]
    triangles = find_triangles(obs["station_ids"], n_stations)
    quadrangles = find_quadrangles(obs["station_ids"], n_stations)

    # Compute closure quantities from corrupted visibilities
    vis = obs["vis_corrupt"]
    cphase = compute_closure_phases(vis, obs["station_ids"], triangles)
    logcamp = compute_log_closure_amplitudes(vis, obs["station_ids"], quadrangles)
    sigma_cp = closure_phase_sigma(obs["sigma_vis"], vis, obs["station_ids"], triangles)
    sigma_lca = closure_amplitude_sigma(obs["sigma_vis"], vis, obs["station_ids"], quadrangles)

    closure = {
        "triangles": triangles,
        "quadrangles": quadrangles,
        "cphase_rad": cphase,
        "logcamp": logcamp,
        "sigma_cp_rad": sigma_cp,
        "sigma_lca": sigma_lca,
    }

    return obs, closure, meta
