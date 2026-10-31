"""
Synthetic data generation for photoacoustic tomography.

Generates spherical targets, simulates PA signals via the forward model,
and saves data in the standardised npz format.
"""

import json
import numpy as np
import os

from .physics_model import simulate_pa_signals, generate_ground_truth_image


def define_targets():
    """Define 4 spherical targets: 3 small + 1 large.

    Returns
    -------
    tar_info : np.ndarray, shape (4, 4)
        Each row is [x, y, z, radius] in meters.
    z_target : float
        Target plane z-coordinate in meters.
    """
    z_target_mm = 15.0  # mm
    # 3 small targets (1.5 mm diameter = 0.75 mm radius) + 1 large (5 mm diameter = 2.5 mm radius)
    tar_info = 1e-3 * np.array([
        [-5.0, 0.0, z_target_mm, 0.75],
        [5.0, 0.0, z_target_mm, 0.75],
        [0.0, 5.0, z_target_mm, 0.75],
        [0.0, -5.0, z_target_mm, 2.5],
    ])
    return tar_info, z_target_mm * 1e-3


def define_detector_array():
    """Define 31x31 planar detector array.

    Returns
    -------
    xd : np.ndarray, shape (31,)
        X-coordinates of detector centres in meters.
    yd : np.ndarray, shape (31,)
        Y-coordinates of detector centres (same as xd).
    """
    aperture_len = 20e-3  # 20 mm aperture
    det_pitch = (2.0 / 3.0) * 1e-3  # 2/3 mm pitch
    xd = np.arange(-aperture_len / 2, aperture_len / 2 + det_pitch, det_pitch)
    yd = xd.copy()
    return xd, yd


def define_time_vector():
    """Define sampling time vector.

    Returns
    -------
    t : np.ndarray, shape (n_time,)
        Time vector in seconds.
    fs : float
        Sampling frequency in Hz.
    """
    fs = 20e6  # 20 MHz
    ts = 1.0 / fs
    t = np.arange(0, 65e-6 + ts, ts)
    return t, fs


def generate_and_save(data_dir="data", c=1484.0):
    """Generate synthetic PA data and save to npz files.

    Parameters
    ----------
    data_dir : str
        Output directory.
    c : float
        Sound speed in m/s.
    """
    os.makedirs(data_dir, exist_ok=True)

    tar_info, z_target = define_targets()
    xd, yd = define_detector_array()
    t, fs = define_time_vector()

    print(f"Simulating PA signals for {tar_info.shape[0]} targets "
          f"on {len(xd)}x{len(yd)} detector array...")
    signals = simulate_pa_signals(tar_info, xd, yd, t, c=c)
    print(f"Signal shape: {signals.shape}, mean: {np.mean(signals):.6e}")

    # Save raw data with batch dimension
    np.savez(
        f"{data_dir}/raw_data.npz",
        signals=signals[np.newaxis],          # (1, n_time, n_det_x, n_det_y)
        detector_x=xd[np.newaxis],            # (1, n_det_x)
        detector_y=yd[np.newaxis],            # (1, n_det_y)
        time_vector=t[np.newaxis],            # (1, n_time)
    )

    # Generate ground truth image on the reconstruction grid
    resolution = 500e-6
    xf = np.arange(xd[0], xd[-1] + resolution, resolution)
    yf = xf.copy()
    gt_image = generate_ground_truth_image(tar_info, xf, yf)

    np.savez(
        f"{data_dir}/ground_truth.npz",
        ground_truth_image=gt_image[np.newaxis],  # (1, nx, ny)
        image_x=xf[np.newaxis],                   # (1, nx)
        image_y=yf[np.newaxis],                   # (1, ny)
    )

    # Save metadata (imaging parameters only, no solver params)
    meta = {
        "sound_speed_m_per_s": c,
        "sampling_frequency_hz": fs,
        "aperture_length_m": 20e-3,
        "detector_pitch_m": (2.0 / 3.0) * 1e-3,
        "detector_size_m": 2e-3,
        "num_subdetectors": 25,
        "target_plane_z_m": z_target,
        "num_detectors_x": len(xd),
        "num_detectors_y": len(yd),
        "num_targets": int(tar_info.shape[0]),
        "target_info_description": "Each row: [x_m, y_m, z_m, radius_m]",
    }
    with open(f"{data_dir}/meta_data.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Data saved to {data_dir}/")
    return signals, tar_info, xd, yd, t, z_target


if __name__ == "__main__":
    generate_and_save()
