"""
Photoacoustic forward model.

Simulates photoacoustic signals recorded by a planar detector array from
spherical absorbing targets. Based on Xu & Wang (2005) Eq. 23.
"""

import numpy as np


def step_function(x):
    """Heaviside step function for arrays.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        0 where x < 0, 0.5 where x == 0, 1 where x > 0.
    """
    return 0.5 * (np.sign(x) + 1)


def pa_signal_single_target(tar_info, xd, yd, t, c=1484.0,
                             det_len=2e-3, num_subdet=25):
    """Simulate photoacoustic signals from a single spherical target.

    Each detector is subdivided into sub-elements to simulate spatial
    integration across the detector surface.

    Parameters
    ----------
    tar_info : array_like, shape (4,)
        Target parameters [x, y, z, radius] in meters.
    xd : np.ndarray, shape (n_det_x,)
        X-coordinates of detector centres in meters.
    yd : np.ndarray, shape (n_det_y,)
        Y-coordinates of detector centres in meters.
    t : np.ndarray, shape (n_time,)
        Time vector in seconds.
    c : float
        Sound speed in m/s.
    det_len : float
        Side length of square detector in meters.
    num_subdet : int
        Number of sub-elements per detector (must be a perfect square).

    Returns
    -------
    prec : np.ndarray, shape (n_time, n_det_x, n_det_y)
        Recorded pressure time-series at each detector.
    """
    num_sd_x = int(np.sqrt(num_subdet))
    sd_pitch = det_len / np.sqrt(num_subdet)
    sd_ind = np.arange(num_sd_x) - (num_sd_x - 1) / 2
    sd_ind = sd_ind.astype(int)
    sd_offset = sd_pitch * sd_ind

    prec = np.empty([len(t), len(xd), len(yd)])

    tar_xyz = tar_info[0:3]
    tar_rad = tar_info[3]

    for xi in range(len(xd)):
        for yi in range(len(yd)):
            pa_sig = 0
            for m in range(num_sd_x):
                for n in range(num_sd_x):
                    det_xyz = np.array([xd[xi] + sd_offset[m],
                                        yd[yi] + sd_offset[n], 0.0])
                    R = np.linalg.norm(det_xyz - tar_xyz)
                    st = tar_rad - abs(R - c * t)
                    pa_sig = pa_sig + step_function(st) * (R - c * t) / (2 * R)
            pr = pa_sig / num_subdet
            prec[:, xi, yi] = pr

    return prec


def simulate_pa_signals(tar_info_array, xd, yd, t, c=1484.0,
                         det_len=2e-3, num_subdet=25):
    """Simulate photoacoustic signals from multiple spherical targets.

    Parameters
    ----------
    tar_info_array : np.ndarray, shape (n_targets, 4)
        Each row is [x, y, z, radius] in meters.
    xd : np.ndarray, shape (n_det_x,)
        X-coordinates of detector centres in meters.
    yd : np.ndarray, shape (n_det_y,)
        Y-coordinates of detector centres in meters.
    t : np.ndarray, shape (n_time,)
        Time vector in seconds.
    c : float
        Sound speed in m/s.
    det_len : float
        Side length of square detector in meters.
    num_subdet : int
        Number of sub-elements per detector.

    Returns
    -------
    signals : np.ndarray, shape (n_time, n_det_x, n_det_y)
        Summed pressure time-series from all targets.
    """
    signals = np.zeros([len(t), len(xd), len(yd)])
    for jj in range(tar_info_array.shape[0]):
        signals += pa_signal_single_target(
            tar_info_array[jj, :], xd, yd, t, c, det_len, num_subdet)
    return signals


def generate_ground_truth_image(tar_info_array, xf, yf):
    """Generate binary ground truth image showing target locations.

    Parameters
    ----------
    tar_info_array : np.ndarray, shape (n_targets, 4)
        Each row is [x, y, z, radius] in meters.
    xf : np.ndarray, shape (nx,)
        X-coordinates of image pixels in meters.
    yf : np.ndarray, shape (ny,)
        Y-coordinates of image pixels in meters.

    Returns
    -------
    gt : np.ndarray, shape (nx, ny)
        Binary image: 1 inside targets, 0 outside.
    """
    Yf, Xf = np.meshgrid(yf, xf)
    gt = np.zeros_like(Xf)
    for jj in range(tar_info_array.shape[0]):
        cx, cy, cz, rad = tar_info_array[jj]
        dist = np.sqrt((Xf - cx)**2 + (Yf - cy)**2)
        gt[dist <= rad] = 1.0
    return gt
