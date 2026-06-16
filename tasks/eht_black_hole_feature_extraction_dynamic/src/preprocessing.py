"""
Data Preprocessing for Dynamic α-DPI Feature Extraction
=========================================================

Handles loading per-frame EHT observation data (npz format), computing closure
quantity indices from station IDs, computing NUFFT parameters, and estimating flux.

No ehtim dependency — closure indices are computed directly from station pairs.

Reference
---------
Sun et al. (2022), ApJ 932:99 — α-DPI
"""

import os
import json
import numpy as np
import torch
from itertools import combinations


def load_metadata(data_dir: str = "data") -> dict:
    """
    Load imaging and training parameters from JSON metadata file.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing meta_data.

    Returns
    -------
    dict with all metadata fields.
    """
    path = os.path.join(data_dir, "meta_data.json")
    with open(path, "r") as f:
        return json.load(f)


def load_raw_data(data_dir: str = "data") -> dict:
    """
    Load raw observation data from npz file.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing raw_data.npz.

    Returns
    -------
    dict with npz contents.
    """
    path = os.path.join(data_dir, "raw_data.npz")
    return dict(np.load(path, allow_pickle=False))


def load_frame_data(raw_data: dict, frame_idx: int) -> dict:
    """
    Extract per-frame observation data from the raw_data dict.

    Parameters
    ----------
    raw_data : dict
        Loaded from raw_data.npz.
    frame_idx : int
        Frame index.

    Returns
    -------
    dict with keys:
        'vis'         : (M,) complex ndarray — complex visibilities
        'vis_sigma'   : (M,) ndarray — per-visibility noise std
        'uv_coords'   : (M, 2) ndarray — (u, v) baseline coordinates in wavelengths
        'station_ids' : (M, 2) int ndarray — station pair indices
    """
    return {
        "vis": raw_data[f"vis_{frame_idx}"],
        "vis_sigma": raw_data[f"sigma_{frame_idx}"],
        "uv_coords": raw_data[f"uv_{frame_idx}"],
        "station_ids": raw_data[f"station_ids_{frame_idx}"],
    }


def _find_triangles(station_ids):
    """
    Find all closure phase triangles from baseline station pairs.

    For each set of 3 stations (i, j, k) that form a triangle,
    returns the baseline indices and signs for computing closure phase.

    Parameters
    ----------
    station_ids : (M, 2) int ndarray — station pairs for each baseline

    Returns
    -------
    triangles : list of (idx1, sign1, idx2, sign2, idx3, sign3)
        Each tuple gives the baseline index and sign (+1 or -1)
        for the three baselines forming the triangle.
    """
    # Build lookup: (s1, s2) -> baseline index
    # Sign convention: vis(s1, s2) = V_{s1-s2}
    # If we need V_{s2-s1} = conj(V_{s1-s2}), sign = -1
    baseline_map = {}
    for idx, (s1, s2) in enumerate(station_ids):
        baseline_map[(s1, s2)] = (idx, +1)
        baseline_map[(s2, s1)] = (idx, -1)

    stations = sorted(set(station_ids.ravel()))
    triangles = []

    for i, j, k in combinations(stations, 3):
        # Try to find baselines (i,j), (j,k), (k,i)
        if (i, j) in baseline_map and (j, k) in baseline_map and (k, i) in baseline_map:
            idx1, sign1 = baseline_map[(i, j)]
            idx2, sign2 = baseline_map[(j, k)]
            idx3, sign3 = baseline_map[(k, i)]
            triangles.append((idx1, sign1, idx2, sign2, idx3, sign3))

    return triangles


def _find_quadrangles(station_ids):
    """
    Find all closure amplitude quadrangles from baseline station pairs.

    For 4 stations (i, j, k, l), the closure amplitude is:
        CA = |V_{ij}| |V_{kl}| / (|V_{ik}| |V_{jl}|)

    Parameters
    ----------
    station_ids : (M, 2) int ndarray

    Returns
    -------
    quads : list of (idx_ij, idx_kl, idx_ik, idx_jl)
        Each tuple gives the 4 baseline indices for the quadrangle.
    """
    baseline_map = {}
    for idx, (s1, s2) in enumerate(station_ids):
        key = (min(s1, s2), max(s1, s2))
        baseline_map[key] = idx

    stations = sorted(set(station_ids.ravel()))
    quads = []

    for i, j, k, l in combinations(stations, 4):
        # Check all 6 baselines exist
        pairs = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
        if all(p in baseline_map for p in pairs):
            # CA = |V_ij| * |V_kl| / (|V_ik| * |V_jl|)
            idx_ij = baseline_map[(i, j)]
            idx_kl = baseline_map[(k, l)]
            idx_ik = baseline_map[(i, k)]
            idx_jl = baseline_map[(j, l)]
            quads.append((idx_ij, idx_kl, idx_ik, idx_jl))

    return quads


def extract_closure_indices(frame_data: dict) -> dict:
    """
    Extract closure phase triangle and closure amplitude quadrangle index maps,
    and compute observed closure quantities from the visibility data.

    Parameters
    ----------
    frame_data : dict
        From load_frame_data(), with keys 'vis', 'vis_sigma', 'station_ids'.

    Returns
    -------
    dict with keys:
        'cphase_ind_list'  : list of 3 int64 ndarrays
        'cphase_sign_list' : list of 3 float64 ndarrays
        'camp_ind_list'    : list of 4 int64 ndarrays
        'cphase_data'      : dict with 'cphase' (observed values) and 'sigmacp'
        'camp_data'        : dict with 'camp' (observed values)
        'logcamp_data'     : dict with 'camp' (log CA values) and 'sigmaca'
    """
    vis = frame_data['vis']
    sigma = frame_data['vis_sigma']
    station_ids = frame_data['station_ids']

    # --- Closure phases ---
    triangles = _find_triangles(station_ids)

    cphase_ind_list = [np.zeros(len(triangles), dtype=np.int64) for _ in range(3)]
    cphase_sign_list = [np.zeros(len(triangles), dtype=np.float64) for _ in range(3)]

    cphase_values = np.zeros(len(triangles))
    cphase_sigmas = np.zeros(len(triangles))

    for t, (idx1, s1, idx2, s2, idx3, s3) in enumerate(triangles):
        cphase_ind_list[0][t] = idx1
        cphase_ind_list[1][t] = idx2
        cphase_ind_list[2][t] = idx3
        cphase_sign_list[0][t] = s1
        cphase_sign_list[1][t] = s2
        cphase_sign_list[2][t] = s3

        # Compute observed closure phase (degrees)
        ang1 = np.angle(vis[idx1]) * s1
        ang2 = np.angle(vis[idx2]) * s2
        ang3 = np.angle(vis[idx3]) * s3
        cphase_values[t] = np.degrees(ang1 + ang2 + ang3)

        # Closure phase sigma (approximate: sqrt(sum of angle variances))
        # sigma_phase_i ≈ sigma_i / |V_i|
        amp1 = np.abs(vis[idx1]) + 1e-20
        amp2 = np.abs(vis[idx2]) + 1e-20
        amp3 = np.abs(vis[idx3]) + 1e-20
        sig_rad = np.sqrt(
            (sigma[idx1] / amp1) ** 2
            + (sigma[idx2] / amp2) ** 2
            + (sigma[idx3] / amp3) ** 2
        )
        cphase_sigmas[t] = np.degrees(sig_rad)

    # --- Closure amplitudes ---
    quads = _find_quadrangles(station_ids)

    camp_ind_list = [np.zeros(len(quads), dtype=np.int64) for _ in range(4)]
    camp_values = np.zeros(len(quads))
    logcamp_values = np.zeros(len(quads))
    logcamp_sigmas = np.zeros(len(quads))

    for q, (idx_ij, idx_kl, idx_ik, idx_jl) in enumerate(quads):
        camp_ind_list[0][q] = idx_ij
        camp_ind_list[1][q] = idx_kl
        camp_ind_list[2][q] = idx_ik
        camp_ind_list[3][q] = idx_jl

        # CA = |V_ij| * |V_kl| / (|V_ik| * |V_jl|)
        amp_ij = np.abs(vis[idx_ij]) + 1e-20
        amp_kl = np.abs(vis[idx_kl]) + 1e-20
        amp_ik = np.abs(vis[idx_ik]) + 1e-20
        amp_jl = np.abs(vis[idx_jl]) + 1e-20

        ca = (amp_ij * amp_kl) / (amp_ik * amp_jl)
        camp_values[q] = ca
        logcamp_values[q] = np.log(ca)

        # Log closure amplitude sigma
        # sigma_logCA ≈ sqrt(sum of (sigma/|V|)^2)
        logcamp_sigmas[q] = np.sqrt(
            (sigma[idx_ij] / amp_ij) ** 2
            + (sigma[idx_kl] / amp_kl) ** 2
            + (sigma[idx_ik] / amp_ik) ** 2
            + (sigma[idx_jl] / amp_jl) ** 2
        )

    return {
        "cphase_ind_list": cphase_ind_list,
        "cphase_sign_list": cphase_sign_list,
        "camp_ind_list": camp_ind_list,
        "cphase_data": {
            "cphase": cphase_values,
            "sigmacp": cphase_sigmas,
        },
        "camp_data": {
            "camp": camp_values,
        },
        "logcamp_data": {
            "camp": logcamp_values,
            "sigmaca": logcamp_sigmas,
        },
    }


def compute_nufft_params(uv_coords: np.ndarray, npix: int,
                         fov_uas: float) -> dict:
    """
    Compute NUFFT trajectory and pulse correction factor for torchkbnufft.

    Parameters
    ----------
    uv_coords : (M, 2) ndarray
        Baseline (u, v) coordinates in wavelengths.
    npix : int
        Image size in pixels.
    fov_uas : float
        Field of view in microarcseconds.

    Returns
    -------
    dict with keys:
        'ktraj_vis'     : (1, 2, M) torch.Tensor — scaled (v, u) trajectory
        'pulsefac_vis'  : (2, M) torch.Tensor — pulse function correction
    """
    uas_to_rad = np.pi / (180.0 * 3600.0 * 1e6)
    fov = fov_uas * uas_to_rad
    psize = fov / npix

    # Note: NUFFT uses (v, u) ordering
    vu = np.hstack((uv_coords[:, 1:2], uv_coords[:, 0:1]))

    pulsefac_amp = (np.sinc(uv_coords[:, 0] * psize) ** 2
                    * np.sinc(uv_coords[:, 1] * psize) ** 2)
    phase = -np.pi * (uv_coords[:, 0] + uv_coords[:, 1]) * psize
    pulsefac_real = pulsefac_amp * np.cos(phase)
    pulsefac_imag = pulsefac_amp * np.sin(phase)

    vu_scaled = np.array(vu * psize * 2 * np.pi)
    ktraj_vis = torch.tensor(vu_scaled.T, dtype=torch.float32).unsqueeze(0)
    pulsefac_vis_torch = torch.tensor(
        np.stack([pulsefac_real, pulsefac_imag], axis=0),
        dtype=torch.float32
    )

    return {
        "ktraj_vis": ktraj_vis,
        "pulsefac_vis": pulsefac_vis_torch,
    }


def estimate_flux(vis: np.ndarray) -> float:
    """
    Estimate total flux from visibility amplitudes.

    Uses the median of all visibility amplitudes as an approximation
    of the zero-spacing flux.

    Parameters
    ----------
    vis : (M,) complex ndarray — complex visibilities

    Returns
    -------
    float — estimated total flux in Jy.
    """
    return float(np.median(np.abs(vis)))


def prepare_frame(raw_data: dict, frame_idx: int, npix: int,
                  fov_uas: float) -> tuple:
    """
    Load and preprocess a single frame's observation data.

    Parameters
    ----------
    raw_data : dict
        Loaded from raw_data.npz via load_raw_data().
    frame_idx : int
        Frame index.
    npix : int
        Image size in pixels.
    fov_uas : float
        Field of view in microarcseconds.

    Returns
    -------
    (frame_data, closure_indices, nufft_params, flux_const)
        frame_data : dict with 'vis', 'vis_sigma', 'uv_coords', 'station_ids'
        closure_indices : dict with closure phase and amplitude index maps
        nufft_params : dict with NUFFT trajectory and pulse correction
        flux_const : float — estimated total flux
    """
    frame_data = load_frame_data(raw_data, frame_idx)
    closure_indices = extract_closure_indices(frame_data)
    nufft_params = compute_nufft_params(frame_data["uv_coords"], npix, fov_uas)
    flux_const = estimate_flux(frame_data["vis"])

    return frame_data, closure_indices, nufft_params, flux_const
