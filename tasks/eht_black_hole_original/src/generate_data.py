"""
Generate Synthetic EHT Dataset with Gain Corruption
=====================================================

Creates a synthetic M87*-like black hole dataset with realistic
station-based gain errors, so that closure quantities differ from
calibrated visibilities.

Telescope properties (location, SEFD) are based on the EHT 2017
campaign.  UV coordinates are computed using astropy for proper
Earth-rotation synthesis (hour-angle via Greenwich Mean Sidereal Time,
WGS84 ellipsoid baseline geometry).  Per-baseline thermal noise is
derived from station SEFDs.

Usage
-----
    python -c "from src.generate_data import generate_dataset; generate_dataset()"

Output files (in data/)
-----------------------
    raw_data.npz    all arrays for reconstruction
    meta_data       JSON imaging parameters
"""

import sys
import os
import json
import numpy as np

from astropy.time import Time
import astropy.units as u


# ═══════════════════════════════════════════════════════════════════════════
# EHT 2017 telescope array
# ═══════════════════════════════════════════════════════════════════════════

EHT_TELESCOPES = {
    "ALMA": {"x_m":  2225061.164, "y_m": -5440057.370, "z_m": -2481681.150, "sefd_jy":   90},
    "APEX": {"x_m":  2225039.530, "y_m": -5441197.630, "z_m": -2479303.360, "sefd_jy": 3500},
    "JCMT": {"x_m": -5464584.680, "y_m": -2493001.170, "z_m":  2150653.980, "sefd_jy": 6000},
    "SMA":  {"x_m": -5464523.400, "y_m": -2493147.080, "z_m":  2150611.750, "sefd_jy": 4900},
    "SMT":  {"x_m": -1828796.200, "y_m": -5054406.800, "z_m":  3427865.200, "sefd_jy": 5000},
    "LMT":  {"x_m":  -768713.964, "y_m": -5988541.798, "z_m":  2063275.947, "sefd_jy":  600},
    "PV":   {"x_m":  5088967.900, "y_m":  -301681.600, "z_m":  3825015.800, "sefd_jy": 1400},
    "SPT":  {"x_m":        0.010, "y_m":        0.010, "z_m": -6359609.700, "sefd_jy": 5000},
}
"""EHT 2017 station properties at 230 GHz.

Source: https://github.com/achael/eht-imaging/blob/main/arrays/EHT2017.txt

Keys
----
x_m, y_m, z_m : float
    Geocentric Cartesian (ITRS/ECEF) coordinates in metres.
sefd_jy : float
    System Equivalent Flux Density in Janskys (lower = more sensitive).
    SEFD values are per-polarisation (Stokes RR or LL).
"""


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic image
# ═══════════════════════════════════════════════════════════════════════════

def make_ring_image(
    N: int = 64,
    ring_radius_frac: float = 0.22,
    ring_width_frac: float = 0.055,
    asymmetry: float = 0.5,
    asymmetry_angle_deg: float = 220.0,
) -> np.ndarray:
    """
    Synthetic black hole ring image (M87*-like morphology).

    Parameters
    ----------
    N : int
        Image size (N x N pixels).
    ring_radius_frac : float
        Ring radius as fraction of image half-width.
    ring_width_frac : float
        Ring Gaussian half-width as fraction of image half-width.
    asymmetry : float
        Brightness contrast (0 = uniform, 1 = strong asymmetry).
    asymmetry_angle_deg : float
        Position angle of brightest arc (degrees).

    Returns
    -------
    image : (N, N) ndarray, normalised so image.sum() = 1.
    """
    coords = np.linspace(-1.0, 1.0, N)
    xx, yy = np.meshgrid(coords, coords)

    r = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)

    phi0 = np.deg2rad(asymmetry_angle_deg)
    brightness_mod = 1.0 + asymmetry * np.cos(theta - phi0)
    brightness_mod = np.maximum(brightness_mod, 0.0)

    ring = (
        np.exp(-((r - ring_radius_frac) ** 2) / (2.0 * ring_width_frac ** 2))
        * brightness_mod
    )

    ring = np.maximum(ring, 0.0)
    ring /= ring.sum()
    return ring


# ═══════════════════════════════════════════════════════════════════════════
# UV-coverage via astropy
# ═══════════════════════════════════════════════════════════════════════════

def simulate_eht_uv_coverage(
    source_ra_deg: float = 187.7059,
    source_dec_deg: float = 12.3911,
    obs_start_utc: str = "2017-04-06T00:00:00",
    obs_duration_hours: float = 6.0,
    n_time_steps: int = 15,
    freq_ghz: float = 230.0,
) -> tuple:
    """
    Simulate EHT uv-coverage using astropy for proper coordinate transforms.

    Station positions are given in ITRS/ECEF Cartesian coordinates (from
    eht-imaging EHT2017.txt).  Hour angles are computed via astropy's
    Greenwich Mean Sidereal Time.

    Parameters
    ----------
    source_ra_deg : float
        Right ascension of the source in degrees (default: M87*).
    source_dec_deg : float
        Declination of the source in degrees.
    obs_start_utc : str
        UTC start time of the observation (ISO-8601).
    obs_duration_hours : float
        Total observation duration in hours.
    n_time_steps : int
        Number of time samples across the observation.
    freq_ghz : float
        Observing frequency in GHz.

    Returns
    -------
    uv_coords : (M, 2) ndarray
        Baseline (u, v) coordinates in wavelengths.
    station_ids : (M, 2) int ndarray
        Station pair indices for each baseline measurement.
    n_stations : int
        Number of stations in the array.
    timestamps : (M,) ndarray
        Modified Julian Date for each visibility sample.
    sefds : (n_stations,) ndarray
        SEFD in Janskys for each station.
    station_names : list of str
        Station name for each index.
    """
    wavelength = 3e8 / (freq_ghz * 1e9)  # metres
    dec = np.deg2rad(source_dec_deg)
    ra_rad = np.deg2rad(source_ra_deg)

    # --- Station positions (ECEF from EHT2017.txt) -----------------------
    names = list(EHT_TELESCOPES.keys())
    n_stations = len(names)
    sefds = np.array([EHT_TELESCOPES[n]["sefd_jy"] for n in names], dtype=np.float64)

    xyz = np.array([
        [EHT_TELESCOPES[n]["x_m"], EHT_TELESCOPES[n]["y_m"], EHT_TELESCOPES[n]["z_m"]]
        for n in names
    ])  # (n_stations, 3)

    # --- Observation times ------------------------------------------------
    t_start = Time(obs_start_utc, scale="utc")
    dt_hours = np.linspace(0, obs_duration_hours, n_time_steps)
    times = t_start + dt_hours * u.hour  # Time array

    # Greenwich Mean Sidereal Time → hour angle
    gmst_rad = np.array([
        t.sidereal_time("mean", "greenwich").rad for t in times
    ])
    ha_arr = gmst_rad - ra_rad  # Greenwich Hour Angle of the source

    # --- Compute baselines and project to (u, v) -------------------------
    uv_list = []
    sid_list = []
    ts_list = []

    for i in range(n_stations):
        for j in range(i + 1, n_stations):
            B = xyz[j] - xyz[i]  # baseline vector (ITRS, metres)
            for k, ha in enumerate(ha_arr):
                u_val = (np.sin(ha) * B[0] + np.cos(ha) * B[1]) / wavelength
                v_val = (
                    -np.sin(dec) * np.cos(ha) * B[0]
                    + np.sin(dec) * np.sin(ha) * B[1]
                    + np.cos(dec) * B[2]
                ) / wavelength
                uv_list.append([u_val, v_val])
                sid_list.append([i, j])
                ts_list.append(times[k].mjd)

    uv_coords = np.array(uv_list)
    station_ids = np.array(sid_list, dtype=np.int64)
    timestamps = np.array(ts_list)

    return uv_coords, station_ids, n_stations, timestamps, sefds, names


# ═══════════════════════════════════════════════════════════════════════════
# Per-baseline thermal noise from SEFDs
# ═══════════════════════════════════════════════════════════════════════════

def compute_sefd_noise(
    station_ids: np.ndarray,
    sefds: np.ndarray,
    eta: float = 0.88,
    bandwidth_hz: float = 2e9,
    tau_int: float = 10.0,
) -> np.ndarray:
    """
    Per-baseline thermal noise from station SEFDs.

    σ_ij = (1/η) sqrt(SEFD_i SEFD_j) / sqrt(2 Δν τ)

    Parameters
    ----------
    station_ids : (M, 2) int ndarray
    sefds : (n_stations,) ndarray  — SEFD in Jy
    eta : float — quantization efficiency (0.88 for 2-bit)
    bandwidth_hz : float — recording bandwidth in Hz
    tau_int : float — integration time in seconds

    Returns
    -------
    noise_std_per_vis : (M,) ndarray — 1-σ thermal noise per visibility (Jy)
    """
    sefd_i = sefds[station_ids[:, 0]]
    sefd_j = sefds[station_ids[:, 1]]
    return (1.0 / eta) * np.sqrt(sefd_i * sefd_j) / np.sqrt(2.0 * bandwidth_hz * tau_int)


def apply_station_gains(
    vis_clean: np.ndarray,
    station_ids: np.ndarray,
    n_stations: int,
    amp_error: float = 0.2,
    phase_error_deg: float = 30.0,
    rng=None,
) -> tuple:
    """
    Apply station-based gain errors to visibilities.

    Each station i gets a complex gain g_i. The corrupted visibility is:
        V_ij^corr = g_i * conj(g_j) * V_ij^true

    Gain amplitudes: |g_i| ~ 1 + U(-amp_error, amp_error)
    Gain phases: arg(g_i) ~ U(-phase_error, phase_error)

    Parameters
    ----------
    vis_clean : (M,) complex — true visibilities
    station_ids : (M, 2) int
    n_stations : int
    amp_error : float — fractional amplitude error (e.g. 0.2 = 20%)
    phase_error_deg : float — max phase error in degrees
    rng : numpy random Generator

    Returns
    -------
    vis_corrupted : (M,) complex
    gains : (n_stations,) complex — applied gains
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate station gains
    amp = 1.0 + rng.uniform(-amp_error, amp_error, n_stations)
    phase = rng.uniform(-np.deg2rad(phase_error_deg),
                        np.deg2rad(phase_error_deg), n_stations)
    gains = amp * np.exp(1j * phase)

    # Apply gains: V_ij^corr = g_i * conj(g_j) * V_ij^true
    g_i = gains[station_ids[:, 0]]
    g_j = gains[station_ids[:, 1]]
    vis_corrupted = g_i * np.conj(g_j) * vis_clean

    return vis_corrupted, gains


# ═══════════════════════════════════════════════════════════════════════════
# Forward model (self-contained for data generation)
# ═══════════════════════════════════════════════════════════════════════════

def _build_measurement_matrix(uv_coords, N, pixel_size_rad):
    """Build DFT measurement matrix A, shape (M, N²)."""
    idx = np.arange(N) - N // 2
    l, m = np.meshgrid(idx * pixel_size_rad, idx * pixel_size_rad)
    l_flat = l.ravel()
    m_flat = m.ravel()
    phase = -2j * np.pi * (
        uv_coords[:, 0:1] * l_flat[np.newaxis, :] +
        uv_coords[:, 1:2] * m_flat[np.newaxis, :]
    )
    return np.exp(phase)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset assembly
# ═══════════════════════════════════════════════════════════════════════════

def generate_dataset(
    N: int = 64,
    pixel_size_uas: float = 2.0,
    gain_amp_error: float = 0.2,
    gain_phase_error_deg: float = 30.0,
    eta: float = 0.88,
    bandwidth_hz: float = 2e9,
    tau_int: float = 10.0,
    seed: int = 42,
    save_dir: str = "data",
) -> dict:
    """
    Generate and save a complete synthetic EHT dataset with gain corruption.

    Per-baseline noise is derived from telescope SEFDs (not a single SNR).

    Parameters
    ----------
    N : int — image size
    pixel_size_uas : float — pixel size in microarcseconds
    gain_amp_error : float — fractional amplitude gain error
    gain_phase_error_deg : float — phase gain error in degrees
    eta : float — quantization efficiency
    bandwidth_hz : float — recording bandwidth in Hz
    tau_int : float — integration time in seconds
    seed : int — random seed
    save_dir : str — output directory

    Returns
    -------
    dict with all dataset arrays
    """
    rng = np.random.default_rng(seed)

    uas_to_rad = np.pi / (180.0 * 3600.0 * 1e6)
    pixel_size_rad = pixel_size_uas * uas_to_rad

    print(f"Generating {N}x{N} synthetic M87* image ...")
    image = make_ring_image(N=N)

    print("Simulating EHT uv-coverage (astropy) ...")
    uv_coords, station_ids, n_stations, timestamps, sefds, station_names = \
        simulate_eht_uv_coverage()
    M = len(uv_coords)
    print(f"  -> {M} baselines, {n_stations} stations")
    for i, name in enumerate(station_names):
        t = EHT_TELESCOPES[name]
        print(f"     {name:>5s}  SEFD = {t['sefd_jy']:>5.0f} Jy")

    print("Computing visibilities ...")
    A = _build_measurement_matrix(uv_coords, N, pixel_size_rad)
    vis_true = A @ image.ravel()

    print("Adding SEFD-based thermal noise ...")
    noise_std_per_vis = compute_sefd_noise(
        station_ids, sefds, eta=eta, bandwidth_hz=bandwidth_hz, tau_int=tau_int,
    )
    noise = noise_std_per_vis * (
        rng.standard_normal(M) + 1j * rng.standard_normal(M)
    ) / np.sqrt(2)
    vis_true_noisy = vis_true + noise
    median_noise = float(np.median(noise_std_per_vis))
    print(f"  noise range: {noise_std_per_vis.min():.4e} – {noise_std_per_vis.max():.4e} Jy")
    print(f"  median noise: {median_noise:.4e} Jy")

    print(f"Applying gain errors (amp={gain_amp_error:.0%}, phase={gain_phase_error_deg} deg) ...")
    vis_corrupted, gains = apply_station_gains(
        vis_true_noisy, station_ids, n_stations,
        amp_error=gain_amp_error,
        phase_error_deg=gain_phase_error_deg,
        rng=rng,
    )

    dataset = dict(
        image=image,
        vis_true=vis_true_noisy,
        vis_corrupted=vis_corrupted,
        uv_coords=uv_coords,
        station_ids=station_ids,
        noise_std_per_vis=noise_std_per_vis,
        timestamps=timestamps,
        gains=gains,
        sefds=sefds,
        station_names=station_names,
    )

    metadata = {
        "N": N,
        "pixel_size_uas": pixel_size_uas,
        "pixel_size_rad": pixel_size_rad,
        "noise_std": median_noise,
        "freq_ghz": 230.0,
        "source_ra_deg": 187.7059,
        "source_dec_deg": 12.3911,
        "obs_start_utc": "2017-04-06T00:00:00",
        "obs_duration_hours": 6.0,
        "n_baselines": M,
        "n_stations": n_stations,
        "eta": eta,
        "bandwidth_hz": bandwidth_hz,
        "tau_int": tau_int,
        "gain_amp_error": gain_amp_error,
        "gain_phase_error_deg": gain_phase_error_deg,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        np.savez(
            os.path.join(save_dir, "raw_data.npz"),
            vis_corrupted=vis_corrupted,
            vis_true=vis_true_noisy,
            uv_coords=uv_coords,
            station_ids=station_ids,
            noise_std_per_vis=noise_std_per_vis,
            timestamps=timestamps,
        )

        with open(os.path.join(save_dir, "meta_data"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save ground truth for evaluation
        ref_dir = os.path.join(os.path.dirname(save_dir), "evaluation", "reference_outputs")
        os.makedirs(ref_dir, exist_ok=True)
        np.save(os.path.join(ref_dir, "ground_truth.npy"), image)

        print(f"\nDataset saved to '{save_dir}/'")
        print(f"  image shape   : {image.shape}")
        print(f"  baselines     : {M}")
        print(f"  stations      : {n_stations}")
        print(f"  pixel size    : {pixel_size_uas} uas")

    dataset["metadata"] = metadata
    return dataset


if __name__ == "__main__":
    generate_dataset()
