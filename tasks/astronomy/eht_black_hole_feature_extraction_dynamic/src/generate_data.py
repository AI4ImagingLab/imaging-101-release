"""
Dynamic Synthetic Data Generation for α-DPI Feature Extraction
==============================================================

Generates a sequence of time-varying simple crescent images (position angle
rotation) and simulates per-frame EHT 2017 observations.

No ehtim dependency — fully self-contained with numpy, scipy, astropy.
UV coverage is computed via Earth-rotation synthesis with astropy GMST.
Visibilities are computed with a DFT forward model.

Reference
---------
EHT Collaboration (2022), ApJL 930:L15 — Sgr A* Paper IV (Variability)
Sun et al. (2022), ApJ 932:99 — α-DPI
"""

import os
import json
import numpy as np

from astropy.time import Time
import astropy.units as u


# ── EHT 2017 Array Configuration ────────────────────────────────────────────
# 8-station array used for M87 observations in April 2017.
# Coordinates: ITRS/ECEF geocentric Cartesian (meters).
# Source: eht-imaging/arrays/EHT2017.txt

EHT2017_TELESCOPES = {
    "ALMA": {"x_m":  2225061.164, "y_m": -5440057.370, "z_m": -2481681.150,
             "sefd_jy":    90, "code": "AA"},
    "APEX": {"x_m":  2225039.530, "y_m": -5441197.630, "z_m": -2479303.360,
             "sefd_jy":  3500, "code": "AP"},
    "JCMT": {"x_m": -5464584.680, "y_m": -2493001.170, "z_m":  2150653.980,
             "sefd_jy":  6000, "code": "JC"},
    "SMA":  {"x_m": -5464523.400, "y_m": -2493147.080, "z_m":  2150611.750,
             "sefd_jy":  4900, "code": "SM"},
    "SMT":  {"x_m": -1828796.200, "y_m": -5054406.800, "z_m":  3427865.200,
             "sefd_jy":  5000, "code": "AZ"},
    "LMT":  {"x_m":  -768713.964, "y_m": -5988541.798, "z_m":  2063275.947,
             "sefd_jy":   600, "code": "LM"},
    "PV":   {"x_m":  5088967.900, "y_m":  -301681.600, "z_m":  3825015.800,
             "sefd_jy":  1400, "code": "PV"},
    "SPT":  {"x_m":        0.010, "y_m":        0.010, "z_m": -6359609.700,
             "sefd_jy":  5000, "code": "SP"},
}


def generate_simple_crescent_image(
        npix, fov_uas, diameter_uas, width_uas, asymmetry, pa_deg):
    """
    Generate a simple crescent (asymmetric Gaussian ring) image.

    Uses the same grid convention as SimpleCrescentParam2Img in physics_model.py.

    Parameters
    ----------
    npix : int
        Image size in pixels.
    fov_uas : float
        Field of view in microarcseconds.
    diameter_uas : float
        Ring diameter in microarcseconds.
    width_uas : float
        Ring width (Gaussian sigma) in microarcseconds.
    asymmetry : float
        Brightness asymmetry in [0, 1].
    pa_deg : float
        Position angle in degrees (bright side, E of N).

    Returns
    -------
    (npix, npix) ndarray — normalized image (unit sum).
    """
    half_fov = 0.5 * fov_uas
    eps = 1e-4

    gap = 1.0 / npix
    xs = np.arange(-1 + gap, 1, 2 * gap)
    grid_y, grid_x = np.meshgrid(-xs, xs, indexing='ij')
    grid_r = np.sqrt(grid_x ** 2 + grid_y ** 2)
    grid_theta = np.arctan2(grid_y, grid_x)

    r = (diameter_uas / 2) / half_fov
    sigma = width_uas / half_fov
    eta = pa_deg * np.pi / 180

    ring = np.exp(-0.5 * (grid_r - r) ** 2 / sigma ** 2)
    S = 1 + asymmetry * np.cos(grid_theta - eta)
    image = S * ring
    image = image / (image.sum() + eps)

    return image


# ── UV Coverage via astropy ──────────────────────────────────────────────────

def simulate_eht_uv_coverage_at_time(
    source_ra_deg, source_dec_deg, time_utc, freq_ghz=230.0,
):
    """
    Compute instantaneous EHT uv-coverage for a single time stamp.

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
    uv_coords : (M, 2) ndarray — baseline (u, v) in wavelengths
    station_ids : (M, 2) int ndarray — station pair indices
    sefds : (n_stations,) ndarray — SEFDs in Jy
    """
    wavelength = 3e8 / (freq_ghz * 1e9)
    dec = np.deg2rad(source_dec_deg)
    ra_rad = np.deg2rad(source_ra_deg)

    names = list(EHT2017_TELESCOPES.keys())
    n_stations = len(names)
    sefds = np.array(
        [EHT2017_TELESCOPES[n]["sefd_jy"] for n in names], dtype=np.float64)

    xyz = np.array([
        [EHT2017_TELESCOPES[n]["x_m"],
         EHT2017_TELESCOPES[n]["y_m"],
         EHT2017_TELESCOPES[n]["z_m"]]
        for n in names
    ])

    t = Time(time_utc, scale="utc")
    gmst_rad = t.sidereal_time("mean", "greenwich").rad
    ha = gmst_rad - ra_rad

    uv_list = []
    sid_list = []
    for i in range(n_stations):
        for j in range(i + 1, n_stations):
            B = xyz[j] - xyz[i]
            u_val = (np.sin(ha) * B[0] + np.cos(ha) * B[1]) / wavelength
            v_val = (
                -np.sin(dec) * np.cos(ha) * B[0]
                + np.sin(dec) * np.sin(ha) * B[1]
                + np.cos(dec) * B[2]
            ) / wavelength
            uv_list.append([u_val, v_val])
            sid_list.append([i, j])

    return (np.array(uv_list), np.array(sid_list, dtype=np.int64), sefds)


def compute_sefd_noise(station_ids, sefds, bandwidth_hz=2e9,
                       tau_int=10.0, eta=0.88):
    """
    Per-baseline thermal noise from station SEFDs.

    sigma_ij = (1/eta) sqrt(SEFD_i * SEFD_j) / sqrt(2 * BW * tau_int)
    """
    sefd_i = sefds[station_ids[:, 0]]
    sefd_j = sefds[station_ids[:, 1]]
    return ((1.0 / eta) * np.sqrt(sefd_i * sefd_j)
            / np.sqrt(2.0 * bandwidth_hz * tau_int))


def build_dft_matrix(uv_coords, N, pixel_size_rad):
    """
    Build DFT measurement matrix A, shape (M, N*N).

    A[k, p] = exp(+2pi i (u_k l_p + v_k m_p))

    Uses ehtim convention for pixel coordinates and torchkbnufft sign convention
    (positive exponent) to match physics_model.py NUFFT forward model.
    """
    xlist = (np.arange(0, -N, -1) * pixel_size_rad
             + (pixel_size_rad * N) / 2.0 - pixel_size_rad / 2.0)
    ylist = (np.arange(0, -N, -1) * pixel_size_rad
             + (pixel_size_rad * N) / 2.0 - pixel_size_rad / 2.0)
    l, m = np.meshgrid(xlist, ylist)
    l_flat = l.ravel()
    m_flat = m.ravel()
    phase = 2j * np.pi * (
        uv_coords[:, 0:1] * l_flat[np.newaxis, :]
        + uv_coords[:, 1:2] * m_flat[np.newaxis, :]
    )
    return np.exp(phase)


# ── Dataset generation ───────────────────────────────────────────────────────

def generate_dynamic_dataset(
        n_frames: int = 10,
        obs_duration_hr: float = 8.0,
        npix: int = 64,
        fov_uas: float = 120.0,
        diameter_uas: float = 44.0,
        width_uas: float = 8.0,
        asymmetry: float = 0.5,
        pa_start_deg: float = -130.0,
        pa_end_deg: float = -50.0,
        total_flux: float = 0.6,
        sys_noise: float = 0.02,
        source_ra_deg: float = 187.70593,   # M87: RA = 12h 30m 49.42s
        source_dec_deg: float = 12.391123,   # M87: Dec = +12d 23' 28.0"
        freq_ghz: float = 230.0,
        bandwidth_hz: float = 2e9,
        tau_int: float = 10.0,
        obs_start_utc: str = "2017-04-06T00:00:00",
        seed: int = 42,
        save_dir: str = "data") -> dict:
    """
    Generate a time-varying crescent observation sequence with EHT 2017 array.

    The crescent position angle rotates linearly from pa_start_deg to pa_end_deg
    over n_frames frames, simulating Sgr A*-like variability. Each frame covers
    a separate time snapshot.

    Data is saved as a single npz file (no ehtim dependency required).

    Parameters
    ----------
    n_frames : int
        Number of time snapshots.
    obs_duration_hr : float
        Total observation duration in hours.
    npix : int
        Image size in pixels.
    fov_uas : float
        Field of view in microarcseconds.
    diameter_uas : float
        Ring diameter in microarcseconds (constant over time).
    width_uas : float
        Ring width in microarcseconds (constant over time).
    asymmetry : float
        Brightness asymmetry [0, 1] (constant over time).
    pa_start_deg : float
        Position angle at first frame (degrees, E of N).
    pa_end_deg : float
        Position angle at last frame (degrees, E of N).
    total_flux : float
        Total flux in Jy.
    sys_noise : float
        Fractional systematic noise added to visibilities.
    source_ra_deg : float
        Right ascension of the source in degrees (default: M87).
    source_dec_deg : float
        Declination of the source in degrees.
    freq_ghz : float
        Observing frequency in GHz.
    bandwidth_hz : float
        Recording bandwidth in Hz.
    tau_int : float
        Integration time in seconds.
    obs_start_utc : str
        UTC start time of the observation (ISO-8601).
    seed : int
        Random seed for reproducibility.
    save_dir : str
        Output directory.

    Returns
    -------
    dict — metadata including per-frame ground truth.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(save_dir, exist_ok=True)

    uas_to_rad = np.pi / (180.0 * 3600.0 * 1e6)
    pixel_size_rad = (fov_uas / npix) * uas_to_rad

    frame_duration_hr = obs_duration_hr / n_frames
    frame_times_hr = np.array([i * frame_duration_hr for i in range(n_frames)])
    pa_values = np.linspace(pa_start_deg, pa_end_deg, n_frames)

    t_start = Time(obs_start_utc, scale="utc")
    station_names = list(EHT2017_TELESCOPES.keys())

    ground_truth_per_frame = []
    save_dict = {
        "n_frames": np.array(n_frames),
        "frame_times": frame_times_hr,
    }

    print(f"Generating {n_frames} frames over {obs_duration_hr} hr "
          f"(PA: {pa_start_deg:.0f} -> {pa_end_deg:.0f} deg)...")

    for i in range(n_frames):
        pa = pa_values[i]

        # Generate crescent image for this frame
        image = generate_simple_crescent_image(
            npix, fov_uas, diameter_uas, width_uas, asymmetry, pa)
        image_flux = image * total_flux  # Jy/pixel

        # Compute UV coverage at the frame's epoch
        frame_time = t_start + frame_times_hr[i] * u.hour
        uv_coords, station_ids, sefds = simulate_eht_uv_coverage_at_time(
            source_ra_deg, source_dec_deg, frame_time.iso, freq_ghz)

        M_t = len(uv_coords)

        # Forward model: image -> visibilities via DFT
        A = build_dft_matrix(uv_coords, npix, pixel_size_rad)
        vis_true = A @ image_flux.ravel()

        # SEFD-based thermal noise
        noise_std = compute_sefd_noise(station_ids, sefds,
                                       bandwidth_hz=bandwidth_hz,
                                       tau_int=tau_int)
        noise = noise_std * (
            rng.standard_normal(M_t) + 1j * rng.standard_normal(M_t)
        ) / np.sqrt(2)
        vis_noisy = vis_true + noise

        # Add fractional systematic noise
        if sys_noise > 0:
            sys_err = sys_noise * np.abs(vis_noisy) * (
                rng.standard_normal(M_t) + 1j * rng.standard_normal(M_t)
            ) / np.sqrt(2)
            vis_noisy = vis_noisy + sys_err
            # Update noise std to include systematic component
            noise_std = np.sqrt(noise_std ** 2
                                + (sys_noise * np.abs(vis_noisy)) ** 2)

        # Store per-frame arrays in save_dict
        save_dict[f"vis_{i}"] = vis_noisy
        save_dict[f"sigma_{i}"] = noise_std
        save_dict[f"uv_{i}"] = uv_coords
        save_dict[f"station_ids_{i}"] = station_ids

        ground_truth_per_frame.append({
            "diameter_uas": diameter_uas,
            "width_uas": width_uas,
            "asymmetry": asymmetry,
            "position_angle_deg": float(pa),
        })

        print(f"  Frame {i:2d}: t={frame_times_hr[i]:.2f}h, "
              f"PA={pa:+.1f} deg, {M_t} baselines, "
              f"median noise={np.median(noise_std):.4e} Jy")

    # Save ground truth frames
    gt_images = []
    for gt in ground_truth_per_frame:
        gt_img = generate_simple_crescent_image(
            npix, fov_uas, gt['diameter_uas'], gt['width_uas'],
            gt['asymmetry'], gt['position_angle_deg'])
        gt_images.append(gt_img)
    save_dict["ground_truth_images"] = np.array(gt_images)

    # Save raw_data.npz
    np.savez(os.path.join(save_dir, "raw_data.npz"), **save_dict)

    # Build and save metadata
    metadata = {
        "npix": npix,
        "fov_uas": fov_uas,
        "pixel_size_uas": fov_uas / npix,
        "n_frames": n_frames,
        "obs_duration_hr": obs_duration_hr,
        "frame_duration_hr": frame_duration_hr,
        "frame_times_hr": frame_times_hr.tolist(),
        "geometric_model": "simple_crescent",
        "n_gaussian": 0,
        "r_range": [10.0, 40.0],
        "width_range": [1.0, 40.0],
        "total_flux": total_flux,
        "freq_ghz": freq_ghz,
        "source_ra_deg": source_ra_deg,
        "source_dec_deg": source_dec_deg,
        "obs_start_utc": obs_start_utc,
        "n_stations": len(station_names),
        "station_names": station_names,
        "bandwidth_hz": bandwidth_hz,
        "tau_int": tau_int,
        # alpha-DPI training config
        "n_flow": 16,
        "n_epoch": 5000,
        "batch_size": 2048,
        "lr": 1e-4,
        "logdet_weight": 1.0,
        "grad_clip": 1e-4,
        "alpha_divergence": 1.0,
        "beta": 0.0,
        "start_order": 4,
        "decay_rate": 1000,
        "seqfrac_inv": 4,
        "data_product": "cphase_logcamp",
        "sys_noise": sys_noise,
        "ground_truth_per_frame": ground_truth_per_frame,
    }

    with open(os.path.join(save_dir, "meta_data"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nDynamic dataset saved to {save_dir}/")
    print(f"  Frames         : {n_frames}")
    print(f"  Duration       : {obs_duration_hr} hr "
          f"({frame_duration_hr * 60:.0f} min/frame)")
    print(f"  PA rotation    : {pa_start_deg:.0f} -> {pa_end_deg:.0f} deg")

    return metadata


if __name__ == "__main__":
    generate_dynamic_dataset()
