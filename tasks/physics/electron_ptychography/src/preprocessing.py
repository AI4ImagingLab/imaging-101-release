"""Preprocessing: load 4D-STEM data, calibrate, and compute derived quantities."""

import json
import os

import numpy as np


def load_data(data_dir):
    """Load 4D-STEM datacube and vacuum probe from raw_data.npz.

    Parameters
    ----------
    data_dir : str
        Path to data/ directory containing raw_data.npz.

    Returns
    -------
    datacube : np.ndarray, float32, shape (Rx, Ry, Qx, Qy)
        4D-STEM dataset.
    probe : np.ndarray, float32, shape (Qx, Qy)
        Vacuum probe intensity.
    """
    raw = np.load(os.path.join(data_dir, "raw_data.npz"))
    datacube = raw["datacube"][0]       # remove batch dim (1, Rx, Ry, Qx, Qy) -> (Rx, Ry, Qx, Qy)
    probe = raw["vacuum_probe"][0]      # remove batch dim (1, Qx, Qy) -> (Qx, Qy)
    return datacube, probe


def load_metadata(data_dir):
    """Load imaging parameters from meta_data.json.

    Returns
    -------
    meta : dict
        Keys: energy_eV, R_pixel_size_A, convergence_semiangle_mrad,
        scan_shape, detector_shape, defocus_A,
        com_rotation_deg.
    """
    with open(os.path.join(data_dir, "meta_data.json")) as f:
        return json.load(f)


def _get_probe_size(dp, thresh_lower=0.01, thresh_upper=0.99, N=100):
    """Estimate probe radius and center from a diffraction pattern.

    Algorithm (from py4DSTEM):
    1. Create N binary masks by thresholding dp at linearly-spaced levels.
    2. Compute radius r = sqrt(area / pi) for each mask.
    3. Find the stable plateau in r(threshold) by taking the derivative
       and selecting where |dr/dthresh| is small.
    4. Average the stable r values to get the probe radius.
    5. Compute the center-of-mass of dp within the corresponding mask.

    Parameters
    ----------
    dp : np.ndarray, shape (Qx, Qy)
    thresh_lower, thresh_upper : float
        Threshold range as fraction of max intensity.
    N : int
        Number of threshold levels.

    Returns
    -------
    r : float
        Probe radius in pixels.
    x0 : float
        Center x coordinate (row).
    y0 : float
        Center y coordinate (column).
    """
    thresh_vals = np.linspace(thresh_lower, thresh_upper, N)
    r_vals = np.zeros(N)
    dp_max = np.max(dp)

    for i, thresh in enumerate(thresh_vals):
        mask = dp > dp_max * thresh
        r_vals[i] = np.sqrt(np.sum(mask) / np.pi)

    # Find stable plateau: derivative is small and non-positive
    dr = np.gradient(r_vals)
    stable = (dr <= 0) & (dr >= 2 * np.median(dr))
    r = np.mean(r_vals[stable])

    # Center of mass using the mean threshold from the stable range
    thresh_mean = np.mean(thresh_vals[stable])
    mask = dp > dp_max * thresh_mean
    ny, nx = dp.shape
    ry, rx = np.meshgrid(np.arange(nx), np.arange(ny))
    masked_dp = dp * mask
    total = np.sum(masked_dp)
    x0 = np.sum(rx * masked_dp) / total
    y0 = np.sum(ry * masked_dp) / total

    return r, x0, y0


def calibrate_datacube(datacube, probe, R_pixel_size, convergence_semiangle,
                       thresh_upper=0.7):
    """Estimate probe size from the vacuum probe measurement.

    Parameters
    ----------
    datacube : np.ndarray, shape (Rx, Ry, Qx, Qy)
        4D-STEM dataset (used only for shape validation).
    probe : np.ndarray
        Vacuum probe intensity, shape (Qx, Qy).
    R_pixel_size : float
        Real-space pixel size in Angstroms.
    convergence_semiangle : float
        Probe convergence semi-angle in milliradians.
    thresh_upper : float
        Upper threshold for probe edge detection.

    Returns
    -------
    probe_radius_pixels : float
    probe_center : tuple of float
        (qx0, qy0) center of the probe in detector pixels.
    """
    probe_radius_pixels, probe_qx0, probe_qy0 = _get_probe_size(
        probe, thresh_upper=thresh_upper,
    )
    return probe_radius_pixels, (probe_qx0, probe_qy0)


def compute_dp_mean(datacube):
    """Compute the mean diffraction pattern.

    Parameters
    ----------
    datacube : np.ndarray, shape (Rx, Ry, Qx, Qy)

    Returns
    -------
    dp_mean : np.ndarray, shape (Qx, Qy)
    """
    return np.mean(datacube, axis=(0, 1))


def compute_virtual_images(datacube, center, radius, expand=2.0):
    """Compute bright-field and dark-field virtual images.

    Parameters
    ----------
    datacube : np.ndarray, shape (Rx, Ry, Qx, Qy)
    center : tuple of float
        (qx0, qy0) probe center.
    radius : float
        Probe radius in pixels.
    expand : float
        Extra pixels beyond the probe radius for the BF disk.

    Returns
    -------
    bf : np.ndarray, shape (Rx, Ry)
    df : np.ndarray, shape (Rx, Ry)
    """
    Qx, Qy = datacube.shape[2], datacube.shape[3]
    qx0, qy0 = center
    radius_bf = radius + expand

    # Distance from center for each detector pixel
    qy_arr, qx_arr = np.meshgrid(np.arange(Qy), np.arange(Qx))
    dist = np.sqrt((qx_arr - qx0) ** 2 + (qy_arr - qy0) ** 2)

    bf_mask = dist <= radius_bf
    df_mask = dist > radius_bf

    # Sum over masked detector pixels
    bf = np.sum(datacube * bf_mask[None, None, :, :], axis=(2, 3))
    df = np.sum(datacube * df_mask[None, None, :, :], axis=(2, 3))

    return bf, df


def compute_bf_mask(dp_mean, threshold=0.8):
    """Threshold mean DP to create bright-field disk mask.

    Parameters
    ----------
    dp_mean : np.ndarray, shape (Qx, Qy)
    threshold : float
        Fraction of maximum intensity.

    Returns
    -------
    mask : np.ndarray, bool, shape (Qx, Qy)
    """
    return dp_mean > threshold
