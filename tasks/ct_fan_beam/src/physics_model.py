"""
Fan-beam CT forward model and filtered back-projection.

Fan-beam geometry:
    - Point X-ray source rotates on a circle of radius D_sd (source-to-isocenter)
    - Flat linear detector opposite the source at distance D_sd + D_dd from source
    - Divergent rays from source through object to detector elements

Forward model adapted from xtie97/CT_fanbeam_recon_numba (ray-tracing).
FBP adapted from leehoy/CTReconstruction (distance-weighted filtering + backprojection).
"""

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Fan-beam geometry helpers
# ---------------------------------------------------------------------------

def fan_beam_geometry(N, n_det, n_angles, D_sd, D_dd, angle_range=2 * np.pi):
    """Compute fan-beam acquisition geometry parameters.

    Parameters
    ----------
    N : int
        Image size (N x N pixels).
    n_det : int
        Number of detector elements.
    n_angles : int
        Number of projection angles.
    D_sd : float
        Source-to-isocenter distance (pixels).
    D_dd : float
        Isocenter-to-detector distance (pixels).
    angle_range : float
        Total angular range in radians (2*pi for full-scan).

    Returns
    -------
    geo : dict
        Geometry parameters.
    """
    det_width = 2.0 * (D_sd + D_dd) * np.tan(np.arctan(N / 2 / D_sd))
    det_spacing = det_width / n_det
    angles = np.linspace(0, angle_range, n_angles, endpoint=False)
    # Detector element positions (centered)
    det_pos = (np.arange(n_det) - (n_det - 1) / 2.0) * det_spacing

    return {
        'N': N,
        'n_det': n_det,
        'n_angles': n_angles,
        'D_sd': D_sd,
        'D_dd': D_dd,
        'det_spacing': det_spacing,
        'det_pos': det_pos,
        'angles': angles,
        'angle_range': angle_range,
    }


# ---------------------------------------------------------------------------
# Forward projection (image -> sinogram) via ray-driven method
# ---------------------------------------------------------------------------

def fan_beam_forward(image, geo):
    """Compute fan-beam sinogram via pixel-driven forward projection.

    For each source position, projects each image pixel onto the detector
    and accumulates its contribution. This is fully vectorized over pixels,
    with only a loop over projection angles.

    Adapted from xtie97/CT_fanbeam_recon_numba.

    Parameters
    ----------
    image : np.ndarray
        2D image, shape (N, N).
    geo : dict
        Geometry from fan_beam_geometry().

    Returns
    -------
    sinogram : np.ndarray
        Fan-beam sinogram, shape (n_angles, n_det).
    """
    return fan_beam_forward_vectorized(image, geo)


def fan_beam_forward_vectorized(image, geo):
    """Pixel-driven fan-beam forward projection (fully vectorized).

    For each angle, computes the detector coordinate that each pixel
    maps to and distributes pixel values via linear interpolation
    (pixel-driven approach). Only loops over angles.

    Parameters
    ----------
    image : np.ndarray
        2D image, shape (N, N).
    geo : dict
        Geometry from fan_beam_geometry().

    Returns
    -------
    sinogram : np.ndarray
        Fan-beam sinogram, shape (n_angles, n_det).
    """
    N = geo['N']
    D_sd = geo['D_sd']
    D_dd = geo['D_dd']
    angles = geo['angles']
    det_pos = geo['det_pos']
    n_angles = len(angles)
    n_det = len(det_pos)

    sinogram = np.zeros((n_angles, n_det), dtype=np.float64)
    half_N = N / 2.0

    # Pixel coordinates (centered)
    x = np.arange(N) - half_N + 0.5
    y = np.arange(N) - half_N + 0.5
    xx, yy = np.meshgrid(x, y)
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()
    img_flat = image.ravel()

    det_min = det_pos[0]
    det_spacing = det_pos[1] - det_pos[0]

    for i_angle in range(n_angles):
        theta = angles[i_angle]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Rotated coordinates for each pixel
        t = xx_flat * cos_t + yy_flat * sin_t
        s = -xx_flat * sin_t + yy_flat * cos_t

        # Fan-beam magnification: project pixel onto detector
        U = D_sd - s  # distance from source to pixel plane
        det_coord = t * (D_sd + D_dd) / U  # magnified detector coordinate

        # Map to detector index (fractional)
        det_idx = (det_coord - det_min) / det_spacing

        # Only accumulate pixels that fall within detector
        valid = (det_idx >= 0) & (det_idx < n_det - 1) & (U > 0)
        if not np.any(valid):
            continue

        idx_v = det_idx[valid]
        img_v = img_flat[valid]

        # Linear interpolation weights (distribute pixel to two bins)
        idx0 = idx_v.astype(int)
        frac = idx_v - idx0

        # Accumulate with path length weighting
        # Weight by (D_sd + D_dd) / U to account for ray divergence
        weight = (D_sd + D_dd) / U[valid]

        np.add.at(sinogram[i_angle], idx0, (1 - frac) * img_v * weight)
        np.add.at(sinogram[i_angle], idx0 + 1, frac * img_v * weight)

    # Scale by pixel size
    sinogram *= 1.0  # pixel size = 1
    return sinogram


# ---------------------------------------------------------------------------
# Fan-beam back-projection (adjoint of forward projection)
# ---------------------------------------------------------------------------

def fan_beam_backproject(sinogram, geo):
    """Fan-beam back-projection (adjoint of forward projection).

    For each pixel, find its projection onto the detector for each angle,
    accumulate weighted contributions.

    Parameters
    ----------
    sinogram : np.ndarray
        Fan-beam sinogram, shape (n_angles, n_det).
    geo : dict
        Geometry from fan_beam_geometry().

    Returns
    -------
    image : np.ndarray
        Back-projected image, shape (N, N).
    """
    N = geo['N']
    D_sd = geo['D_sd']
    D_dd = geo['D_dd']
    angles = geo['angles']
    det_pos = geo['det_pos']
    n_angles = len(angles)
    n_det = len(det_pos)

    half_N = N / 2.0
    x = np.arange(N) - half_N + 0.5
    y = np.arange(N) - half_N + 0.5
    xx, yy = np.meshgrid(x, y)

    image = np.zeros((N, N), dtype=np.float64)
    d_angle = angles[1] - angles[0] if n_angles > 1 else 1.0

    for i_angle in range(n_angles):
        theta = angles[i_angle]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Rotated coordinates
        t = xx * cos_t + yy * sin_t    # along source-detector line
        s = -xx * sin_t + yy * cos_t   # perpendicular

        # Fan-beam magnification: project pixel onto detector
        # Detector coordinate for each pixel
        U = D_sd - s  # distance from source to pixel along source-detector axis
        det_coord = t * (D_sd + D_dd) / U

        # Distance weighting: 1/U^2 (normalized by D_sd^2)
        weight = (D_sd ** 2) / (U ** 2)

        # Interpolate sinogram row
        f = interp1d(det_pos, sinogram[i_angle, :], kind='linear',
                     bounds_error=False, fill_value=0.0)
        image += weight * f(det_coord) * d_angle

    return image


# ---------------------------------------------------------------------------
# Ramp filter construction
# ---------------------------------------------------------------------------

def ramp_filter(n_det, det_spacing, filter_type='ram-lak', cutoff=0.5):
    """Construct ramp filter for fan-beam FBP in frequency domain.

    Adapted from leehoy/CTReconstruction FanBeam.py Filter().

    Parameters
    ----------
    n_det : int
        Number of detector elements.
    det_spacing : float
        Detector element spacing.
    filter_type : str
        Filter window: 'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'.
    cutoff : float
        Cutoff frequency in (0, 0.5].

    Returns
    -------
    filt : np.ndarray
        Filter in frequency domain, length = zero-padded FFT size.
    pad_len : int
        Zero-padded FFT length.
    """
    pad_len = int(2 ** np.ceil(np.log2(2 * (n_det - 1))))
    N_filt = pad_len + 1

    x = np.arange(N_filt) - (N_filt - 1) / 2.0
    h = np.zeros(N_filt)
    h[x == 0] = 1 / (8 * det_spacing ** 2)
    odds = (x % 2 == 1)
    h[odds] = -0.5 / (np.pi * det_spacing * x[odds]) ** 2
    h = h[:-1]

    filt = np.abs(np.fft.fftshift(np.fft.fft(h))) * 2

    w = 2 * np.pi * np.arange(pad_len) / pad_len
    w = w - np.pi  # center
    w = np.fft.fftshift(w)
    w_shifted = np.fft.fftshift(w)

    # Re-compute w for windowing (centered)
    w = 2 * np.pi * (np.arange(pad_len) - pad_len / 2) / pad_len

    if filter_type == 'ram-lak':
        pass
    elif filter_type == 'shepp-logan':
        with np.errstate(divide='ignore', invalid='ignore'):
            window = np.sinc(w / (2 * np.pi * cutoff))
        filt *= np.fft.fftshift(window)
    elif filter_type == 'cosine':
        window = np.cos(w / (2 * cutoff))
        filt *= np.fft.fftshift(window)
    elif filter_type == 'hamming':
        window = 0.54 + 0.46 * np.cos(w / cutoff)
        filt *= np.fft.fftshift(window)
    elif filter_type == 'hann':
        window = 0.5 + 0.5 * np.cos(w / cutoff)
        filt *= np.fft.fftshift(window)

    # Apply cutoff
    cutoff_mask = np.abs(w) > np.pi * cutoff
    filt[np.fft.fftshift(cutoff_mask)] = 0

    return filt, pad_len


# ---------------------------------------------------------------------------
# Parker weighting for short-scan fan-beam
# ---------------------------------------------------------------------------

def parker_weights(angles, det_pos, D_sd):
    """Compute Parker weights for short-scan fan-beam CT.

    Parker weighting smoothly ramps the contribution of redundant data
    in short-scan acquisitions (angle_range = pi + 2*fan_angle).

    Parameters
    ----------
    angles : np.ndarray
        Projection angles in radians, shape (n_angles,).
    det_pos : np.ndarray
        Detector element positions, shape (n_det,).
    D_sd : float
        Source-to-isocenter distance.

    Returns
    -------
    weights : np.ndarray
        Parker weights, shape (n_angles, n_det).
    """
    n_angles = len(angles)
    n_det = len(det_pos)

    # Fan angle for each detector element
    gamma = np.arctan(det_pos / D_sd)  # (n_det,)
    gamma_max = np.max(np.abs(gamma))

    # Total angular range
    delta = angles[-1] - angles[0] + (angles[1] - angles[0])  # total sweep

    weights = np.ones((n_angles, n_det), dtype=np.float64)

    for j in range(n_det):
        g = gamma[j]
        for i in range(n_angles):
            beta = angles[i] - angles[0]

            # Parker smooth weighting function (S-curve)
            epsilon = gamma_max  # transition width

            if beta < 2 * (epsilon + g):
                if (epsilon + g) > 0:
                    x = beta / (2 * (epsilon + g))
                    x = np.clip(x, 0, 1)
                    weights[i, j] = np.sin(np.pi / 2 * x) ** 2
                else:
                    weights[i, j] = 0.0
            elif beta > delta - 2 * (epsilon - g):
                denom = 2 * (epsilon - g)
                if denom > 0:
                    x = (delta - beta) / denom
                    x = np.clip(x, 0, 1)
                    weights[i, j] = np.sin(np.pi / 2 * x) ** 2
                else:
                    weights[i, j] = 0.0

    return weights


# ---------------------------------------------------------------------------
# Fan-beam FBP reconstruction
# ---------------------------------------------------------------------------

def fan_beam_fbp(sinogram, geo, filter_type='hann', cutoff=0.3,
                 short_scan=False):
    """Fan-beam filtered back-projection reconstruction.

    Adapted from leehoy/CTReconstruction FanBeam.py Reconstruction().

    Steps:
    1. (Optional) Apply Parker weights for short-scan
    2. Pre-weight sinogram: w(gamma) = D_sd / sqrt(D_sd^2 + gamma^2)
    3. Filter each row with ramp filter
    4. Back-project with distance weighting 1/U^2

    Parameters
    ----------
    sinogram : np.ndarray
        Fan-beam sinogram, shape (n_angles, n_det).
    geo : dict
        Geometry from fan_beam_geometry().
    filter_type : str
        Filter window type.
    cutoff : float
        Filter cutoff.
    short_scan : bool
        If True, apply Parker weighting for short-scan geometry.

    Returns
    -------
    recon : np.ndarray
        Reconstructed image, shape (N, N).
    """
    N = geo['N']
    D_sd = geo['D_sd']
    D_dd = geo['D_dd']
    angles = geo['angles']
    det_pos = geo['det_pos']
    det_spacing = geo['det_spacing']
    n_angles = len(angles)
    n_det = len(det_pos)

    # Effective detector position scaled to isocenter plane
    gamma_iso = det_pos * D_sd / (D_sd + D_dd)

    # Step 1: Parker weighting for short-scan
    sino = sinogram.copy()
    if short_scan:
        pw = parker_weights(angles, det_pos, D_sd)
        sino *= pw

    # Step 2: Pre-weighting
    pre_weight = D_sd / np.sqrt(D_sd ** 2 + gamma_iso ** 2)
    sino *= pre_weight[np.newaxis, :]

    # Step 3: Ramp filtering
    eff_spacing = det_spacing * D_sd / (D_sd + D_dd)
    filt, pad_len = ramp_filter(n_det, eff_spacing, filter_type, cutoff)

    filtered = np.zeros_like(sino)
    for i in range(n_angles):
        padded = np.fft.fft(sino[i, :], pad_len)
        filtered_row = np.real(np.fft.ifft(
            np.fft.ifftshift(filt * np.fft.fftshift(padded))
        ))
        filtered[i, :] = filtered_row[:n_det]

    # Step 4: Back-projection with 1/U^2 weighting
    half_N = N / 2.0
    x = np.arange(N) - half_N + 0.5
    y = np.arange(N) - half_N + 0.5
    xx, yy = np.meshgrid(x, y)

    recon = np.zeros((N, N), dtype=np.float64)
    d_angle = angles[1] - angles[0] if n_angles > 1 else 1.0

    for i_angle in range(n_angles):
        theta = angles[i_angle]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        t = xx * cos_t + yy * sin_t
        s = -xx * sin_t + yy * cos_t

        # Magnification mapping: pixel -> detector coordinate
        U = D_sd - s
        det_coord = t * (D_sd + D_dd) / U

        # Distance weighting
        weight = (D_sd ** 2) / (U ** 2)

        # Interpolate filtered sinogram
        f = interp1d(det_pos, filtered[i_angle, :], kind='linear',
                     bounds_error=False, fill_value=0.0)
        recon += weight * f(det_coord) * d_angle

    return recon


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

def add_gaussian_noise(sinogram, sigma, rng=None):
    """Add Gaussian noise to sinogram.

    Parameters
    ----------
    sinogram : np.ndarray
        Clean sinogram.
    sigma : float
        Noise standard deviation.
    rng : np.random.Generator or None

    Returns
    -------
    noisy : np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng()
    return sinogram + rng.normal(0, sigma, sinogram.shape)
