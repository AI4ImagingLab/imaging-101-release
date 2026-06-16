"""
Data Preprocessing for DPI (Deep Probabilistic Imaging)
========================================================

Handles loading EHT observation data (raw_data.npz), extracting closure quantity
indices, computing NUFFT parameters, and building the Gaussian prior image.

No ehtim dependency — all preprocessing is pure numpy/torch.

Pipeline: raw_data.npz + ground_truth.npz + meta_data.json → preprocessed arrays for DPI training

Reference
---------
Sun & Bouman (2020), arXiv:2010.14462 — Deep Probabilistic Imaging
"""

import os
import json
import numpy as np
import torch

_HAS_NFFT = False


def load_observation(data_dir: str = "data") -> dict:
    """
    Load EHT observation from raw_data.npz.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing raw_data.npz.

    Returns
    -------
    dict with keys:
        'vis'          : (M,) complex ndarray — complex visibilities
        'vis_sigma'    : (M,) ndarray — per-visibility noise standard deviation
        'uv_coords'    : (M, 2) ndarray — (u, v) baseline coordinates in wavelengths
        'times'        : (M,) float — observation times
        't1'           : (M,) bytes — station 1 names
        't2'           : (M,) bytes — station 2 names
        'station_ids'  : (M, 2) int — integer station pair indices
        'cp_times'     : (N_cp,) float
        'cp_t1/t2/t3'  : (N_cp,) bytes — closure triangle station names
        'cp_values_deg': (N_cp,) float — observed closure phases (degrees)
        'cp_sigmas_deg': (N_cp,) float — closure phase sigmas (degrees)
        'lca_times'    : (N_lca,) float
        'lca_t1/.../t4': (N_lca,) bytes — closure amplitude quad station names
        'lca_values'   : (N_lca,) float — log closure amplitudes
        'lca_sigmas'   : (N_lca,) float — log closure amplitude sigmas
    """
    path = os.path.join(data_dir, "raw_data.npz")
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def load_ground_truth(data_dir: str = "data", npix: int = None,
                       fov_uas: float = None) -> np.ndarray:
    """
    Load ground-truth image from ground_truth.npz.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing ground_truth.npz.
    npix : int, optional
        Ignored (kept for API compatibility). Image size is taken from the file.
    fov_uas : float, optional
        Ignored (kept for API compatibility).

    Returns
    -------
    (npix, npix) ndarray — ground-truth image.
    """
    path = os.path.join(data_dir, "ground_truth.npz")
    return np.load(path)["image"]


def load_metadata(data_dir: str = "data") -> dict:
    """
    Load imaging and training parameters from JSON metadata file.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing meta_data.json.

    Returns
    -------
    dict with all metadata fields (npix, fov_uas, n_flow, n_epoch, etc.)
    """
    path = os.path.join(data_dir, "meta_data.json")
    with open(path, "r") as f:
        return json.load(f)


def extract_closure_indices(obs_data: dict, snrcut: float = 0.0) -> dict:
    """
    Extract closure phase triangle and closure amplitude quadrangle index maps.

    Maps each closure quantity to the corresponding visibility indices and signs,
    enabling efficient GPU computation of closure phases and log closure amplitudes.

    Uses the same zero_symbol trick from the original DPI code to handle the case
    where a visibility index is 0.

    Parameters
    ----------
    obs_data : dict
        From load_observation(), containing raw_data.npz arrays.
    snrcut : float
        Ignored (kept for API compatibility; SNR cut already applied in generate_data.py).

    Returns
    -------
    dict with keys:
        'cphase_ind_list'  : list of 3 int64 ndarrays — visibility indices for closure phase triangles
        'cphase_sign_list' : list of 3 float64 ndarrays — conjugation signs (+1 or -1)
        'camp_ind_list'    : list of 4 int64 ndarrays — visibility indices for closure amplitude quadrangles
        'cphase_data'      : dict with 'cphase' (degrees) and 'sigmacp' (degrees)
        'camp_data'        : dict with 'camp' (linear closure amplitudes)
        'logcamp_data'     : dict with 'camp' (log CA values) and 'sigmaca'
    """
    times = obs_data["times"]
    t1_arr = obs_data["t1"]
    t2_arr = obs_data["t2"]

    # --- Closure phases ---
    cp_times = obs_data["cp_times"]
    cp_t1 = obs_data["cp_t1"]
    cp_t2 = obs_data["cp_t2"]
    cp_t3 = obs_data["cp_t3"]
    cp_values_deg = obs_data["cp_values_deg"]
    cp_sigmas_deg = obs_data["cp_sigmas_deg"]

    n_cp = len(cp_times)
    cphase_map = np.zeros((n_cp, 3))
    zero_symbol = 100000

    for k1 in range(n_cp):
        for k2 in np.where(times == cp_times[k1])[0]:
            if t1_arr[k2] == cp_t1[k1] and t2_arr[k2] == cp_t2[k1]:
                cphase_map[k1, 0] = zero_symbol if k2 == 0 else k2
            elif t2_arr[k2] == cp_t1[k1] and t1_arr[k2] == cp_t2[k1]:
                cphase_map[k1, 0] = -zero_symbol if k2 == 0 else -k2
            elif t1_arr[k2] == cp_t2[k1] and t2_arr[k2] == cp_t3[k1]:
                cphase_map[k1, 1] = zero_symbol if k2 == 0 else k2
            elif t2_arr[k2] == cp_t2[k1] and t1_arr[k2] == cp_t3[k1]:
                cphase_map[k1, 1] = -zero_symbol if k2 == 0 else -k2
            elif t1_arr[k2] == cp_t3[k1] and t2_arr[k2] == cp_t1[k1]:
                cphase_map[k1, 2] = zero_symbol if k2 == 0 else k2
            elif t2_arr[k2] == cp_t3[k1] and t1_arr[k2] == cp_t1[k1]:
                cphase_map[k1, 2] = -zero_symbol if k2 == 0 else -k2

    cphase_ind_list = []
    cphase_sign_list = []
    for col in range(3):
        raw = cphase_map[:, col]
        ind = np.abs(raw).astype(np.int64)
        ind[ind == zero_symbol] = 0
        cphase_ind_list.append(ind)
        cphase_sign_list.append(np.sign(raw))

    # --- Closure amplitudes ---
    lca_times = obs_data["lca_times"]
    lca_t1 = obs_data["lca_t1"]
    lca_t2 = obs_data["lca_t2"]
    lca_t3 = obs_data["lca_t3"]
    lca_t4 = obs_data["lca_t4"]
    lca_values = obs_data["lca_values"]
    lca_sigmas = obs_data["lca_sigmas"]

    n_lca = len(lca_times)
    camp_map = np.zeros((n_lca, 6))
    zero_symbol = 10000

    for k1 in range(n_lca):
        for k2 in np.where(times == lca_times[k1])[0]:
            if t1_arr[k2] == lca_t1[k1] and t2_arr[k2] == lca_t2[k1]:
                camp_map[k1, 0] = zero_symbol if k2 == 0 else k2
            elif t2_arr[k2] == lca_t1[k1] and t1_arr[k2] == lca_t2[k1]:
                camp_map[k1, 0] = -zero_symbol if k2 == 0 else -k2
            elif t1_arr[k2] == lca_t1[k1] and t2_arr[k2] == lca_t3[k1]:
                camp_map[k1, 1] = zero_symbol if k2 == 0 else k2
            elif t2_arr[k2] == lca_t1[k1] and t1_arr[k2] == lca_t3[k1]:
                camp_map[k1, 1] = -zero_symbol if k2 == 0 else -k2
            elif t1_arr[k2] == lca_t1[k1] and t2_arr[k2] == lca_t4[k1]:
                camp_map[k1, 2] = zero_symbol if k2 == 0 else k2
            elif t2_arr[k2] == lca_t1[k1] and t1_arr[k2] == lca_t4[k1]:
                camp_map[k1, 2] = -zero_symbol if k2 == 0 else -k2
            elif t1_arr[k2] == lca_t2[k1] and t2_arr[k2] == lca_t3[k1]:
                camp_map[k1, 3] = zero_symbol if k2 == 0 else k2
            elif t2_arr[k2] == lca_t2[k1] and t1_arr[k2] == lca_t3[k1]:
                camp_map[k1, 3] = -zero_symbol if k2 == 0 else -k2
            elif t1_arr[k2] == lca_t2[k1] and t2_arr[k2] == lca_t4[k1]:
                camp_map[k1, 4] = zero_symbol if k2 == 0 else k2
            elif t2_arr[k2] == lca_t2[k1] and t1_arr[k2] == lca_t4[k1]:
                camp_map[k1, 4] = -zero_symbol if k2 == 0 else -k2
            elif t1_arr[k2] == lca_t3[k1] and t2_arr[k2] == lca_t4[k1]:
                camp_map[k1, 5] = zero_symbol if k2 == 0 else k2
            elif t2_arr[k2] == lca_t3[k1] and t1_arr[k2] == lca_t4[k1]:
                camp_map[k1, 5] = -zero_symbol if k2 == 0 else -k2

    # Columns 0, 5, 2, 3 correspond to baselines (1-2), (3-4), (1-4), (2-3)
    # for log closure amplitude = log|V12| + log|V34| - log|V14| - log|V23|
    camp_ind_list = []
    for col in [0, 5, 2, 3]:
        raw = camp_map[:, col]
        ind = np.abs(raw).astype(np.int64)
        ind[ind == zero_symbol] = 0
        camp_ind_list.append(ind)

    return {
        "cphase_ind_list": cphase_ind_list,
        "cphase_sign_list": cphase_sign_list,
        "camp_ind_list": camp_ind_list,
        "cphase_data": {
            "cphase": cp_values_deg,
            "sigmacp": cp_sigmas_deg,
        },
        "camp_data": {
            "camp": np.exp(lca_values),  # linear closure amplitudes
        },
        "logcamp_data": {
            "camp": lca_values,
            "sigmaca": lca_sigmas,
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
        'pulsefac_vis'  : (2, M) torch.Tensor — pulse function correction [real; imag]
    """
    uas_to_rad = np.pi / (180.0 * 3600.0 * 1e6)
    fov = fov_uas * uas_to_rad
    psize = fov / npix

    uv = uv_coords
    # Note: NUFFT uses (v, u) ordering
    vu = np.hstack((uv[:, 1:2], uv[:, 0:1]))

    # Analytical sinc² pulse correction for trianglePulse2D
    pulsefac_amp = (np.sinc(uv[:, 0] * psize) ** 2
                    * np.sinc(uv[:, 1] * psize) ** 2)
    phase = -np.pi * (uv[:, 0] + uv[:, 1]) * psize
    pulsefac_real = pulsefac_amp * np.cos(phase)
    pulsefac_imag = pulsefac_amp * np.sin(phase)

    # Scale trajectory for torchkbnufft: vu * psize * 2π
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


def build_prior_image(vis: np.ndarray, npix: int, fov_uas: float,
                       prior_fwhm_uas: float = 50.0) -> tuple:
    """
    Build Gaussian prior image for the MEM regularizer.

    The prior is a circular Gaussian centered on the image, with flux set
    to the median absolute visibility amplitude.

    Parameters
    ----------
    vis : (M,) complex ndarray — complex visibilities
    npix : int
        Image size in pixels.
    fov_uas : float
        Field of view in microarcseconds.
    prior_fwhm_uas : float
        FWHM of the Gaussian prior in microarcseconds.

    Returns
    -------
    (prior_image, flux_const) : (ndarray, float)
        prior_image : (npix, npix) ndarray — prior image for MEM
        flux_const  : float — estimated total flux
    """
    uas_to_rad = np.pi / (180.0 * 3600.0 * 1e6)
    fov = fov_uas * uas_to_rad
    psize = fov / npix
    prior_fwhm = prior_fwhm_uas * uas_to_rad

    flux_const = float(np.median(np.abs(vis)))

    # Centered Gaussian prior
    sigma = prior_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    coords = (np.arange(npix) - npix / 2 + 0.5) * psize
    xx, yy = np.meshgrid(coords, coords)
    gauss1 = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    gauss1 *= flux_const / gauss1.sum()

    # Tiny shifted Gaussian floor (epsilon component, as in ehtim add_gauss)
    shift = prior_fwhm
    gauss2 = np.exp(-((xx - shift)**2 + (yy - shift)**2) / (2.0 * sigma**2))
    gauss2 *= flux_const * 1e-6 / gauss2.sum()

    prior_image = gauss1 + gauss2
    return prior_image, flux_const


def prepare_data(data_dir: str = "data") -> tuple:
    """
    Combined loader for all preprocessed DPI inputs.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    (obs_data, closure_indices, nufft_params, prior_image, flux_const, metadata)
    """
    metadata = load_metadata(data_dir)
    npix = metadata["npix"]
    fov_uas = metadata["fov_uas"]
    prior_fwhm_uas = metadata["prior_fwhm_uas"]

    obs_data = load_observation(data_dir)

    closure_indices = extract_closure_indices(obs_data)
    nufft_params = compute_nufft_params(obs_data["uv_coords"], npix, fov_uas)
    prior_image, flux_const = build_prior_image(
        obs_data["vis"], npix, fov_uas, prior_fwhm_uas)

    return obs_data, closure_indices, nufft_params, prior_image, flux_const, metadata
