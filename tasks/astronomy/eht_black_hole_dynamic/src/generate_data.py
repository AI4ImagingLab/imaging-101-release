"""
Generate Synthetic EHT Dynamic Imaging Dataset
===============================================

Creates a synthetic SgrA*-like rotating crescent video observed by the
EHT 2017 array over time.  Each frame gets its own UV-coverage computed
via Earth-rotation synthesis at the frame's epoch, with per-baseline
thermal noise derived from station SEFDs.

No ehtim dependency — fully self-contained with numpy, scipy, astropy.

Telescope properties (location, SEFD) are based on the EHT 2017
campaign.  UV coordinates are computed using astropy for proper
Earth-rotation synthesis (hour-angle via Greenwich Mean Sidereal Time,
WGS84 ellipsoid baseline geometry).

Usage
-----
    python -c "from src.generate_data import generate_dataset; generate_dataset()"

Output files
------------
    data/raw_data.npz              per-frame visibilities, UV coords, noise
    data/meta_data                 JSON imaging parameters
    evaluation/reference_outputs/ground_truth.npy   ground-truth frames
"""

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
# Synthetic ring frame
# ═══════════════════════════════════════════════════════════════════════════

def make_ring_frame(
    N: int = 30,
    ring_radius_frac: float = 0.22,
    ring_width_frac: float = 0.055,
    asymmetry: float = 0.5,
    asymmetry_angle_deg: float = 220.0,
) -> np.ndarray:
    """
    Synthetic black hole ring image (single frame, SgrA*-like morphology).

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
        Position angle of brightest arc (degrees, measured
        counter-clockwise from the positive x-axis).

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
# Rotating crescent video
# ═══════════════════════════════════════════════════════════════════════════

def make_rotating_video(
    N: int = 30,
    n_frames: int = 12,
    rotation_deg: float = 90.0,
    total_flux: float = 2.0,
    base_angle_deg: float = 220.0,
) -> tuple:
    """
    Generate a video of a rotating crescent ring.

    The bright-spot position angle rotates linearly from *base_angle_deg*
    through *base_angle_deg + rotation_deg* over *n_frames* frames.

    Parameters
    ----------
    N : int
        Image size (N x N pixels).
    n_frames : int
        Number of time frames.
    rotation_deg : float
        Total rotation of the bright spot over the video (degrees).
    total_flux : float
        Total flux per frame in Jy.
    base_angle_deg : float
        Starting position angle of the bright arc (degrees).

    Returns
    -------
    frames : list of (N, N) ndarray
        Each frame in Jy/pixel (frame.sum() == total_flux).
    angles : (n_frames,) ndarray
        Position angle of the bright spot for each frame (degrees).
    """
    angles = np.linspace(base_angle_deg, base_angle_deg + rotation_deg, n_frames)
    frames = []
    for angle in angles:
        frame = make_ring_frame(N=N, asymmetry_angle_deg=angle)
        # make_ring_frame returns normalised (sum=1); scale to total_flux
        frame *= total_flux
        frames.append(frame)
    return frames, angles


# ═══════════════════════════════════════════════════════════════════════════
# Per-frame UV-coverage via astropy (Earth rotation synthesis)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_eht_uv_coverage_at_time(
    source_ra_deg: float,
    source_dec_deg: float,
    time_utc: str,
    freq_ghz: float = 230.0,
) -> tuple:
    """
    Compute the instantaneous EHT uv-coverage for a single time stamp.

    Parameters
    ----------
    source_ra_deg : float
        Right ascension of the source in degrees.
    source_dec_deg : float
        Declination of the source in degrees.
    time_utc : str
        UTC time of the observation (ISO-8601).
    freq_ghz : float
        Observing frequency in GHz.

    Returns
    -------
    uv_coords : (M, 2) ndarray
        Baseline (u, v) coordinates in wavelengths.
    station_ids : (M, 2) int ndarray
        Station pair indices for each baseline measurement.
    sefds : (n_stations,) ndarray
        SEFD in Janskys for each station.
    """
    wavelength = 3e8 / (freq_ghz * 1e9)  # metres
    dec = np.deg2rad(source_dec_deg)
    ra_rad = np.deg2rad(source_ra_deg)

    # --- Station positions (ECEF from EHT2017.txt) -----------------------
    names = list(EHT_TELESCOPES.keys())
    n_stations = len(names)
    sefds = np.array(
        [EHT_TELESCOPES[n]["sefd_jy"] for n in names], dtype=np.float64
    )

    xyz = np.array([
        [EHT_TELESCOPES[n]["x_m"],
         EHT_TELESCOPES[n]["y_m"],
         EHT_TELESCOPES[n]["z_m"]]
        for n in names
    ])  # (n_stations, 3)

    # --- Hour angle via Greenwich Mean Sidereal Time ---------------------
    t = Time(time_utc, scale="utc")
    gmst_rad = t.sidereal_time("mean", "greenwich").rad
    ha = gmst_rad - ra_rad

    # --- Compute baselines and project to (u, v) -------------------------
    uv_list = []
    sid_list = []

    for i in range(n_stations):
        for j in range(i + 1, n_stations):
            B = xyz[j] - xyz[i]  # baseline vector (ITRS, metres)
            u_val = (np.sin(ha) * B[0] + np.cos(ha) * B[1]) / wavelength
            v_val = (
                -np.sin(dec) * np.cos(ha) * B[0]
                + np.sin(dec) * np.sin(ha) * B[1]
                + np.cos(dec) * B[2]
            ) / wavelength
            uv_list.append([u_val, v_val])
            sid_list.append([i, j])

    uv_coords = np.array(uv_list)
    station_ids = np.array(sid_list, dtype=np.int64)

    return uv_coords, station_ids, sefds


# ═══════════════════════════════════════════════════════════════════════════
# Per-baseline thermal noise from SEFDs
# ═══════════════════════════════════════════════════════════════════════════

def compute_sefd_noise(
    station_ids: np.ndarray,
    sefds: np.ndarray,
    bandwidth_hz: float = 2e9,
    tau_int: float = 10.0,
    eta: float = 0.88,
) -> np.ndarray:
    """
    Per-baseline thermal noise from station SEFDs.

    sigma_ij = (1/eta) sqrt(SEFD_i * SEFD_j) / sqrt(2 * bandwidth * tau_int)

    Parameters
    ----------
    station_ids : (M, 2) int ndarray
        Station pair indices for each baseline.
    sefds : (n_stations,) ndarray
        SEFD in Jy for each station.
    bandwidth_hz : float
        Recording bandwidth in Hz.
    tau_int : float
        Integration time in seconds.
    eta : float
        Quantization efficiency (0.88 for 2-bit).

    Returns
    -------
    noise_std_per_vis : (M,) ndarray — 1-sigma thermal noise per visibility (Jy).
    """
    sefd_i = sefds[station_ids[:, 0]]
    sefd_j = sefds[station_ids[:, 1]]
    return (1.0 / eta) * np.sqrt(sefd_i * sefd_j) / np.sqrt(2.0 * bandwidth_hz * tau_int)


# ═══════════════════════════════════════════════════════════════════════════
# Forward model (DFT measurement matrix)
# ═══════════════════════════════════════════════════════════════════════════

def build_dft_matrix(
    uv_coords: np.ndarray,
    N: int,
    pixel_size_rad: float,
) -> np.ndarray:
    """
    Build the DFT measurement matrix A, shape (M, N*N).

    A[k, p] = exp(-2pi i (u_k l_p + v_k m_p))

    where (l_p, m_p) are sky coordinates of pixel p and (u_k, v_k) are
    baseline coordinates in wavelengths.

    Parameters
    ----------
    uv_coords : (M, 2) ndarray
        Baseline (u, v) coordinates in wavelengths.
    N : int
        Image size (N x N pixels).
    pixel_size_rad : float
        Angular size of each pixel in radians.

    Returns
    -------
    A : (M, N*N) complex ndarray
        DFT measurement matrix.
    """
    # Use ehtim convention: coordinates count down from (N/2 - 0.5)*psize
    # This matches DFTForwardModel in physics_model.py exactly.
    xlist = np.arange(0, -N, -1) * pixel_size_rad + \
            (pixel_size_rad * N) / 2.0 - pixel_size_rad / 2.0
    ylist = np.arange(0, -N, -1) * pixel_size_rad + \
            (pixel_size_rad * N) / 2.0 - pixel_size_rad / 2.0
    l, m = np.meshgrid(xlist, ylist)
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
    save_dir: str = "data",
    N: int = 30,
    pixel_size_uas: float = 3.4,
    n_frames: int = 12,
    obs_duration_hours: float = 6.0,
    total_flux: float = 2.0,
    rotation_deg: float = 90.0,
    base_angle_deg: float = 220.0,
    source_ra_deg: float = 266.41684,   # SgrA*: RA = 17h 45m 40.04s
    source_dec_deg: float = -28.99219,  # SgrA*: Dec = -29d 00' 28.1"
    obs_start_utc: str = "2017-04-06T00:00:00",
    freq_ghz: float = 230.0,
    eta: float = 0.88,
    bandwidth_hz: float = 2e9,
    tau_int: float = 10.0,
    seed: int = 42,
) -> dict:
    """
    Generate and save a complete synthetic EHT dynamic imaging dataset.

    Creates a rotating crescent ring observed by the EHT 2017 array.
    Each frame gets its own UV-coverage computed at the frame's epoch,
    with per-baseline SEFD-based thermal noise.

    Parameters
    ----------
    save_dir : str
        Output directory for data files.
    N : int
        Image size (N x N pixels).
    pixel_size_uas : float
        Pixel size in microarcseconds.
    n_frames : int
        Number of time frames.
    obs_duration_hours : float
        Total observation window in hours.
    total_flux : float
        Total flux per frame in Jy.
    rotation_deg : float
        Total bright-spot rotation over the observation (degrees).
    base_angle_deg : float
        Starting position angle of the bright arc (degrees).
    source_ra_deg : float
        Right ascension of the source in degrees (default: SgrA*).
    source_dec_deg : float
        Declination of the source in degrees.
    obs_start_utc : str
        UTC start time of the observation (ISO-8601).
    freq_ghz : float
        Observing frequency in GHz.
    eta : float
        Quantization efficiency.
    bandwidth_hz : float
        Recording bandwidth in Hz.
    tau_int : float
        Integration time in seconds.
    seed : int
        Random seed for noise generation.

    Returns
    -------
    dict with all dataset arrays and metadata.
    """
    rng = np.random.default_rng(seed)

    uas_to_rad = np.pi / (180.0 * 3600.0 * 1e6)
    pixel_size_rad = pixel_size_uas * uas_to_rad

    # --- Generate ground-truth video -------------------------------------
    print(f"Generating {n_frames}-frame rotating crescent ({N}x{N} pixels) ...")
    frames, angles = make_rotating_video(
        N=N,
        n_frames=n_frames,
        rotation_deg=rotation_deg,
        total_flux=total_flux,
        base_angle_deg=base_angle_deg,
    )
    frames_gt = np.array(frames)  # (n_frames, N, N)

    # --- Frame times (evenly spaced over observation window) -------------
    frame_hours = np.linspace(0, obs_duration_hours, n_frames)

    t_start = Time(obs_start_utc, scale="utc")

    # --- Per-frame UV-coverage and visibilities --------------------------
    print("Computing per-frame UV-coverage and visibilities ...")
    station_names = list(EHT_TELESCOPES.keys())

    # Per-frame arrays (variable length across frames, stored individually)
    all_vis = []
    all_sigma = []
    all_uv = []
    all_sids = []

    for t_idx in range(n_frames):
        frame_time = t_start + frame_hours[t_idx] * u.hour
        time_iso = frame_time.iso

        # UV-coverage at this instant
        uv_coords, station_ids, sefds = simulate_eht_uv_coverage_at_time(
            source_ra_deg=source_ra_deg,
            source_dec_deg=source_dec_deg,
            time_utc=time_iso,
            freq_ghz=freq_ghz,
        )
        M_t = len(uv_coords)

        # Forward model: image -> visibilities
        A = build_dft_matrix(uv_coords, N, pixel_size_rad)
        vis_true = A @ frames_gt[t_idx].ravel()

        # SEFD-based thermal noise
        noise_std = compute_sefd_noise(
            station_ids, sefds,
            bandwidth_hz=bandwidth_hz, tau_int=tau_int, eta=eta,
        )
        noise = noise_std * (
            rng.standard_normal(M_t) + 1j * rng.standard_normal(M_t)
        ) / np.sqrt(2)
        vis_noisy = vis_true + noise

        all_vis.append(vis_noisy)
        all_sigma.append(noise_std)
        all_uv.append(uv_coords)
        all_sids.append(station_ids)

        print(
            f"  Frame {t_idx:2d}/{n_frames}: "
            f"t={frame_hours[t_idx]:.2f}h, "
            f"M={M_t} baselines, "
            f"angle={angles[t_idx]:.1f} deg, "
            f"median noise={np.median(noise_std):.4e} Jy"
        )

    # --- Assemble output dict --------------------------------------------
    dataset = dict(
        n_frames=n_frames,
        frame_times=frame_hours,
        frames_gt=frames_gt,
        angles=angles,
        station_names=station_names,
        sefds=sefds,
    )
    for t_idx in range(n_frames):
        dataset[f"vis_{t_idx}"] = all_vis[t_idx]
        dataset[f"sigma_{t_idx}"] = all_sigma[t_idx]
        dataset[f"uv_{t_idx}"] = all_uv[t_idx]
        dataset[f"station_ids_{t_idx}"] = all_sids[t_idx]

    # --- Metadata --------------------------------------------------------
    metadata = {
        "N": N,
        "pixel_size_uas": pixel_size_uas,
        "pixel_size_rad": pixel_size_rad,
        "n_frames": n_frames,
        "obs_duration_hours": obs_duration_hours,
        "total_flux": total_flux,
        "rotation_deg": rotation_deg,
        "base_angle_deg": base_angle_deg,
        "freq_ghz": freq_ghz,
        "source_ra_deg": source_ra_deg,
        "source_dec_deg": source_dec_deg,
        "source_name": "SgrA*",
        "obs_start_utc": obs_start_utc,
        "n_stations": len(station_names),
        "station_names": station_names,
        "eta": eta,
        "bandwidth_hz": bandwidth_hz,
        "tau_int": tau_int,
        "baselines_per_frame": len(all_uv[0]),
    }
    dataset["metadata"] = metadata

    # --- Save to disk ----------------------------------------------------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Build save dict for npz — use allow_pickle=False-friendly flat keys
        save_dict = dict(
            n_frames=np.array(n_frames),
            frame_times=frame_hours,
            frames_gt=frames_gt,
        )
        for t_idx in range(n_frames):
            save_dict[f"vis_{t_idx}"] = all_vis[t_idx]
            save_dict[f"sigma_{t_idx}"] = all_sigma[t_idx]
            save_dict[f"uv_{t_idx}"] = all_uv[t_idx]
            save_dict[f"station_ids_{t_idx}"] = all_sids[t_idx]

        np.savez(
            os.path.join(save_dir, "raw_data.npz"),
            **save_dict,
        )

        with open(os.path.join(save_dir, "meta_data"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save ground truth for evaluation
        ref_dir = os.path.join(
            os.path.dirname(save_dir), "evaluation", "reference_outputs"
        )
        os.makedirs(ref_dir, exist_ok=True)
        np.save(os.path.join(ref_dir, "ground_truth.npy"), frames_gt)

        print(f"\nDataset saved to '{save_dir}/'")
        print(f"  frames        : {n_frames}")
        print(f"  image size    : {N}x{N}")
        print(f"  pixel size    : {pixel_size_uas} uas")
        print(f"  total flux    : {total_flux} Jy")
        print(f"  rotation      : {rotation_deg} deg over {obs_duration_hours}h")
        print(f"  baselines/frm : {len(all_uv[0])}")
        print(f"  ground truth  : {ref_dir}/ground_truth.npy")

    return dataset


if __name__ == "__main__":
    generate_dataset()
