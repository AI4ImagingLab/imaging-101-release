"""Solvers: DPC, parallax, and ptychographic reconstruction.

All algorithms implemented in pure numpy/scipy without py4DSTEM dependency.
"""

import numpy as np
from scipy.linalg import polar
from scipy.ndimage import shift as ndimage_shift


# ---------------------------------------------------------------------------
# DPC (Differential Phase Contrast)
# ---------------------------------------------------------------------------

def _compute_com_field(datacube, mask=None):
    """Compute center-of-mass of each diffraction pattern.

    Parameters
    ----------
    datacube : np.ndarray, shape (Rx, Ry, Qx, Qy)
    mask : np.ndarray, bool, optional, shape (Qx, Qy)

    Returns
    -------
    com_x, com_y : np.ndarray, shape (Rx, Ry)
    """
    Rx, Ry, Qx, Qy = datacube.shape
    qx = np.arange(Qx, dtype=np.float64)
    qy = np.arange(Qy, dtype=np.float64)
    QY, QX = np.meshgrid(qy, qx)

    if mask is not None:
        QX_m = QX * mask
        QY_m = QY * mask
    else:
        QX_m = QX
        QY_m = QY

    com_x = np.zeros((Rx, Ry), dtype=np.float64)
    com_y = np.zeros((Rx, Ry), dtype=np.float64)

    for rx in range(Rx):
        for ry in range(Ry):
            dp = datacube[rx, ry].astype(np.float64)
            if mask is not None:
                dp_m = dp * mask
            else:
                dp_m = dp
            total = dp_m.sum()
            if total > 0:
                com_x[rx, ry] = np.sum(dp_m * QX_m) / total
                com_y[rx, ry] = np.sum(dp_m * QY_m) / total

    return com_x, com_y


def _fourier_integrate(com_x, com_y, dx, dy):
    """Integrate a 2D gradient field via Fourier methods.

    Solves for phase phi such that grad(phi) ~ (com_x, com_y).

    Parameters
    ----------
    com_x, com_y : np.ndarray, shape (Rx, Ry)
        Gradient components.
    dx, dy : float
        Pixel spacings.

    Returns
    -------
    phase : np.ndarray, shape (Rx, Ry)
    """
    Nx, Ny = com_x.shape
    kx = np.fft.fftfreq(Nx, d=dx)
    ky = np.fft.fftfreq(Ny, d=dy)
    KY, KX = np.meshgrid(ky, kx)

    k2 = KX ** 2 + KY ** 2
    k2[0, 0] = np.inf

    kx_op = -1j * 0.25 * KX / k2
    ky_op = -1j * 0.25 * KY / k2

    F_cx = np.fft.fft2(com_x)
    F_cy = np.fft.fft2(com_y)

    phase = np.real(np.fft.ifft2(F_cx * kx_op + F_cy * ky_op))
    return phase


def solve_dpc(datacube, energy, dp_mask, com_rotation, R_pixel_size=1.0,
              max_iter=64, stopping_criterion=1e-6):
    """DPC phase reconstruction via center-of-mass integration.

    Parameters
    ----------
    datacube : np.ndarray, shape (Rx, Ry, Qx, Qy)
    energy : float
        Beam energy in eV.
    dp_mask : np.ndarray, bool
        BF disk mask, shape (Qx, Qy).
    com_rotation : float
        Rotation angle in degrees (including any 180-degree flip).
    R_pixel_size : float
        Real-space pixel size in Angstroms.
    max_iter : int
        Maximum iterations for iterative integration.
    stopping_criterion : float
        Stop when step size drops below this.

    Returns
    -------
    phase : np.ndarray, shape (Rx, Ry)
        Reconstructed phase image.
    """
    com_x, com_y = _compute_com_field(datacube, mask=dp_mask)
    com_x -= np.mean(com_x)
    com_y -= np.mean(com_y)

    # Rotate CoM to align with scan coordinates
    angle_rad = np.deg2rad(com_rotation)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    com_rx = cos_a * com_x - sin_a * com_y
    com_ry = sin_a * com_x + cos_a * com_y

    # Iterative Fourier integration with step-size halving
    dx = dy = R_pixel_size
    phase = np.zeros_like(com_rx)
    step = 0.5
    prev_error = np.inf

    for iteration in range(max_iter):
        phase_dx = (np.roll(phase, 1, axis=0) - np.roll(phase, -1, axis=0)) / (2 * dx)
        phase_dy = (np.roll(phase, 1, axis=1) - np.roll(phase, -1, axis=1)) / (2 * dy)
        res_x = phase_dx - com_rx
        res_y = phase_dy - com_ry
        error = np.mean(res_x ** 2 + res_y ** 2)

        if error > prev_error:
            step /= 2
            if step < stopping_criterion:
                break
        prev_error = error

        update = _fourier_integrate(res_x, res_y, dx, dy)
        phase += step * update

    return phase


# ---------------------------------------------------------------------------
# Parallax Reconstruction
# ---------------------------------------------------------------------------

def _cross_correlate_shift(img1, img2):
    """Estimate integer-pixel shift between two images via cross-correlation."""
    F1 = np.fft.fft2(img1)
    F2 = np.fft.fft2(img2)
    cc = np.abs(np.fft.ifft2(F1 * np.conj(F2)))
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    Nx, Ny = cc.shape
    sx = peak[0] if peak[0] <= Nx // 2 else peak[0] - Nx
    sy = peak[1] if peak[1] <= Ny // 2 else peak[1] - Ny
    return float(sx), float(sy)


def solve_parallax(datacube, energy, com_rotation, R_pixel_size=1.0,
                   transpose=False, threshold_intensity=0.6,
                   fit_aberrations_max_order=3):
    """Parallax phase reconstruction with aberration correction.

    Simplified implementation: aligns virtual BF images via cross-correlation,
    sums the aligned images, and estimates defocus from the shift pattern.

    Parameters
    ----------
    datacube : np.ndarray, shape (Rx, Ry, Qx, Qy)
    energy : float
        Beam energy in eV.
    com_rotation : float
        Rotation angle in degrees (including 180-degree flip).
    R_pixel_size : float
        Real-space pixel size in Angstroms.
    transpose : bool
        Whether to transpose diffraction intensities.
    threshold_intensity : float
        Threshold for selecting BF pixels (fraction of max).
    fit_aberrations_max_order : int
        Maximum radial/angular order for aberration fitting.

    Returns
    -------
    phase : np.ndarray
        Aligned BF image (approximate phase).
    aberrations : dict
        Keys: 'C1' (defocus in Angstrom), 'rotation_Q_to_R_rads', 'transpose'.
    """
    Rx, Ry, Qx, Qy = datacube.shape
    dp_mean = np.mean(datacube, axis=(0, 1))
    bf_mask = dp_mean > threshold_intensity * dp_mean.max()
    bf_pixels = np.argwhere(bf_mask)

    qx0 = np.mean(bf_pixels[:, 0])
    qy0 = np.mean(bf_pixels[:, 1])

    center_idx = np.argmin((bf_pixels[:, 0] - qx0) ** 2 +
                           (bf_pixels[:, 1] - qy0) ** 2)
    cq = bf_pixels[center_idx]
    ref_img = datacube[:, :, cq[0], cq[1]].astype(np.float64)

    shifts_x = np.zeros(len(bf_pixels))
    shifts_y = np.zeros(len(bf_pixels))
    for i, (qxi, qyi) in enumerate(bf_pixels):
        vimg = datacube[:, :, qxi, qyi].astype(np.float64)
        sx, sy = _cross_correlate_shift(ref_img, vimg)
        shifts_x[i] = sx
        shifts_y[i] = sy

    # Aligned BF sum
    aligned_sum = np.zeros((Rx, Ry), dtype=np.float64)
    for i, (qxi, qyi) in enumerate(bf_pixels):
        vimg = datacube[:, :, qxi, qyi].astype(np.float64)
        sx = int(round(shifts_x[i]))
        sy = int(round(shifts_y[i]))
        aligned_sum += np.roll(np.roll(vimg, -sx, axis=0), -sy, axis=1)
    aligned_sum /= len(bf_pixels)

    # Estimate defocus from shift pattern via least-squares
    dq = bf_pixels.astype(np.float64) - np.array([qx0, qy0])
    A = np.column_stack([dq[:, 0], dq[:, 1]])
    Mx, _, _, _ = np.linalg.lstsq(A, shifts_x, rcond=None)
    My, _, _, _ = np.linalg.lstsq(A, shifts_y, rcond=None)
    M = np.array([[Mx[0], Mx[1]], [My[0], My[1]]])

    R_mat, S = polar(M, side="right")
    rotation_rad = -np.arctan2(R_mat[1, 0], R_mat[0, 0])
    C1 = np.mean(np.diag(S)) * R_pixel_size

    aberrations = {
        "C1": C1,
        "rotation_Q_to_R_rads": np.deg2rad(com_rotation),
        "transpose": transpose,
    }

    return aligned_sum, aberrations


# ---------------------------------------------------------------------------
# Ptychography (Single-Slice, Gradient Descent)
# ---------------------------------------------------------------------------

def _electron_wavelength(energy_eV):
    """Relativistic electron wavelength in Angstroms."""
    m0 = 9.10938e-31
    e = 1.60218e-19
    h = 6.62607e-34
    c = 2.99792e8
    E = energy_eV * e
    lam_m = h / np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c ** 2)))
    return lam_m * 1e10


def _build_probe(vacuum_probe_intensity, defocus, energy, R_pixel_size):
    """Build complex probe from vacuum intensity and defocus.

    The vacuum probe intensity is measured in diffraction space.
    We apply a defocus aberration phase and transform to real space.

    Parameters
    ----------
    vacuum_probe_intensity : np.ndarray, shape (Qx, Qy)
    defocus : float
        Defocus in Angstroms.
    energy : float
        Beam energy in eV.
    R_pixel_size : float
        Real-space pixel size in Angstroms.

    Returns
    -------
    probe : np.ndarray, complex, shape (Qx, Qy)
        Complex probe in real space.
    """
    wavelength = _electron_wavelength(energy)
    Qx, Qy = vacuum_probe_intensity.shape

    # Center the vacuum probe via CoM
    qy_arr, qx_arr = np.meshgrid(np.arange(Qy), np.arange(Qx))
    vp = vacuum_probe_intensity.astype(np.float64)
    total = np.sum(vp)
    cx = np.sum(qx_arr * vp) / total
    cy = np.sum(qy_arr * vp) / total

    # Sub-pixel shift to center using Fourier shift
    shift_x = Qx / 2 - cx
    shift_y = Qy / 2 - cy
    kx = np.fft.fftfreq(Qx)
    ky = np.fft.fftfreq(Qy)
    KY, KX = np.meshgrid(ky, kx)
    shift_phase = np.exp(-2j * np.pi * (KX * shift_x + KY * shift_y))
    centered = np.real(np.fft.ifft2(np.fft.fft2(vp) * shift_phase))
    centered = np.maximum(centered, 0)

    # Amplitude in diffraction space (centered)
    probe_amp_k = np.sqrt(centered)

    # Defocus phase: chi(k) = -pi * lambda * defocus * |k|^2
    # k in inverse Angstroms, using real-space sampling
    dk_x = 1.0 / (Qx * R_pixel_size)
    dk_y = 1.0 / (Qy * R_pixel_size)
    kx_phys = np.fft.fftfreq(Qx, d=R_pixel_size)
    ky_phys = np.fft.fftfreq(Qy, d=R_pixel_size)
    KY_phys, KX_phys = np.meshgrid(ky_phys, kx_phys)
    k2 = KX_phys ** 2 + KY_phys ** 2
    chi = -np.pi * wavelength * defocus * k2

    # Probe in Fourier space: centered amplitude * defocus phase
    probe_k = np.fft.fftshift(probe_amp_k) * np.exp(1j * chi)

    # Transform to real space
    probe = np.fft.ifft2(probe_k)
    return probe


def solve_ptychography(datacube, probe_intensity, energy, defocus,
                       com_rotation, transpose=False, max_iter=10,
                       step_size=0.5, batch_fraction=4, seed=None,
                       R_pixel_size=1.0):
    """Single-slice ptychographic reconstruction via gradient descent.

    Parameters
    ----------
    datacube : np.ndarray, shape (Rx, Ry, Qx, Qy)
    probe_intensity : np.ndarray, shape (Qx, Qy)
        Vacuum probe intensity.
    energy : float
        Beam energy in eV.
    defocus : float
        Initial defocus estimate in Angstroms (positive = overfocus).
    com_rotation : float
        Rotation angle in degrees.
    transpose : bool
        Whether to transpose diffraction intensities.
    max_iter : int
        Number of iterations.
    step_size : float
        Gradient descent step size.
    batch_fraction : int
        Divide total patterns by this for mini-batch size.
    seed : int, optional
        Random seed for reproducibility.
    R_pixel_size : float
        Real-space pixel size in Angstroms.

    Returns
    -------
    object_phase : np.ndarray
        Phase of the reconstructed complex object (cropped to scan FOV).
    object_complex : np.ndarray
        Complex object (cropped to scan FOV).
    probe_complex : np.ndarray
        Reconstructed complex probe.
    error_history : list of float
        NMSE at each iteration.
    """
    rng = np.random.default_rng(seed)
    Rx, Ry, Qx, Qy = datacube.shape
    J = Rx * Ry
    batch_size = max(1, J // batch_fraction)

    # Build initial probe
    probe = _build_probe(probe_intensity, defocus, energy, R_pixel_size)
    probe = probe.astype(np.complex128)

    # Measured amplitudes: sqrt of diffraction intensities
    amplitudes = datacube.reshape(J, Qx, Qy).astype(np.float64)
    if transpose:
        amplitudes = np.transpose(amplitudes, (0, 2, 1))
    measured_amp = np.sqrt(np.maximum(amplitudes, 0))
    mean_intensity = np.mean(amplitudes)

    # Normalize probe to match mean diffraction intensity
    probe_k_intensity = np.mean(np.abs(np.fft.fft2(probe)) ** 2)
    if probe_k_intensity > 0:
        probe *= np.sqrt(mean_intensity / probe_k_intensity)

    # Object: must be large enough that every scan position + probe fits
    # Position j maps to obj[j : j+Qx, ...], so we need Rx-1 + Qx rows
    Nx = Rx + Qx - 1
    Ny = Ry + Qy - 1
    obj = np.ones((Nx, Ny), dtype=np.complex128)

    # Scan positions: position (rx, ry) extracts obj[rx:rx+Qx, ry:ry+Qy]
    positions = np.zeros((J, 2), dtype=int)
    idx = 0
    for rx in range(Rx):
        for ry in range(Ry):
            positions[idx] = [rx, ry]
            idx += 1

    eps = 1e-16
    error_history = []

    for iteration in range(max_iter):
        order = rng.permutation(J)
        iteration_error = 0.0

        for batch_start in range(0, J, batch_size):
            batch_end = min(batch_start + batch_size, J)
            batch_indices = order[batch_start:batch_end]

            obj_grad = np.zeros_like(obj)
            probe_grad = np.zeros_like(probe)
            obj_weight = np.zeros((Nx, Ny), dtype=np.float64)
            probe_weight = np.zeros((Qx, Qy), dtype=np.float64)

            for idx in batch_indices:
                r, c = positions[idx]
                obj_patch = obj[r:r + Qx, c:c + Qy]

                exit_wave = probe * obj_patch
                far_field = np.fft.fft2(exit_wave)

                amp_pred = np.abs(far_field)
                amp_meas = measured_amp[idx]
                iteration_error += np.sum((amp_meas - amp_pred) ** 2)

                # Replace amplitude, keep phase
                far_field_mod = amp_meas * np.exp(1j * np.angle(far_field))
                exit_wave_mod = np.fft.ifft2(far_field_mod)
                delta = exit_wave_mod - exit_wave

                # Accumulate gradients
                obj_grad[r:r + Qx, c:c + Qy] += np.conj(probe) * delta
                obj_weight[r:r + Qx, c:c + Qy] += np.abs(probe) ** 2
                probe_grad += np.conj(obj_patch) * delta
                probe_weight += np.abs(obj_patch) ** 2

            # Apply updates
            obj_denom = np.sqrt(eps + obj_weight ** 2)
            obj += step_size * obj_grad / obj_denom
            probe_denom = np.sqrt(eps + probe_weight ** 2)
            probe += step_size * probe_grad / probe_denom

        nmse = iteration_error / (mean_intensity * J * Qx * Qy)
        error_history.append(float(nmse))

    # Crop: remove probe-width border on each side
    border = Qx // 4
    r0 = max(0, border)
    c0 = max(0, border)
    r1 = min(Nx, Nx - border)
    c1 = min(Ny, Ny - border)
    obj_cropped = obj[r0:r1, c0:c1]

    return np.angle(obj_cropped), obj_cropped.copy(), probe.copy(), error_history
