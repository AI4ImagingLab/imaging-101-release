"""
Synthetic Dynamic DCE-MRI Data Generation
==========================================

Creates a time-varying 128x128 phantom with contrast agent uptake,
generates per-frame variable-density Cartesian undersampled k-space,
and adds complex Gaussian noise.

Phantom design:
    - Static background: elliptical body + inner structures (Shepp-Logan-like)
    - Dynamic component: two small enhancing regions that follow a
      gamma-variate contrast uptake curve with different timing parameters

The gamma-variate model for concentration is:
    C(t) = A * ((t - t_arrival) / t_peak)^alpha * exp(-alpha * (t - t_arrival) / t_peak)
    for t > t_arrival, else 0.
"""

import os
import json
import numpy as np


def _make_ellipse(shape, center, axes, angle_deg, intensity):
    """Create a filled ellipse on a 2D grid."""
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    yy = yy - center[0]
    xx = xx - center[1]
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xr = cos_a * xx + sin_a * yy
    yr = -sin_a * xx + cos_a * yy
    mask = (xr / axes[1]) ** 2 + (yr / axes[0]) ** 2 <= 1.0
    img = np.zeros(shape, dtype=np.float64)
    img[mask] = intensity
    return img


def gamma_variate(t, A, t_arrival, t_peak, alpha):
    """
    Gamma-variate contrast uptake curve.

    Parameters
    ----------
    t : ndarray
        Time points.
    A : float
        Peak amplitude scaling.
    t_arrival : float
        Bolus arrival time.
    t_peak : float
        Time to peak (relative to arrival).
    alpha : float
        Shape parameter controlling rise/decay.

    Returns
    -------
    C : ndarray
        Concentration at each time point.
    """
    C = np.zeros_like(t, dtype=np.float64)
    active = t > t_arrival
    dt = (t[active] - t_arrival) / t_peak
    C[active] = A * dt ** alpha * np.exp(-alpha * (dt - 1.0))
    return C


def make_dynamic_phantom(N=128, T=20):
    """
    Create a T-frame dynamic phantom of size (T, N, N).

    The base phantom is a simplified Shepp-Logan-like structure.
    Two small regions exhibit dynamic contrast enhancement following
    gamma-variate curves with different timing parameters.

    Parameters
    ----------
    N : int
        Image size (NxN).
    T : int
        Number of time frames.

    Returns
    -------
    phantom : ndarray, (T, N, N) float64
        Dynamic phantom image series. Values in [0, ~1.5].
    time_points : ndarray, (T,) float64
        Time points in seconds.
    """
    cx, cy = N / 2, N / 2

    # Static base phantom
    base = np.zeros((N, N), dtype=np.float64)
    # Outer ellipse (body)
    base += _make_ellipse((N, N), (cx, cy), (55, 40), 0, 0.4)
    # Inner structure 1
    base += _make_ellipse((N, N), (cx - 10, cy - 8), (18, 14), -20, 0.15)
    # Inner structure 2
    base += _make_ellipse((N, N), (cx + 5, cy + 10), (12, 10), 30, 0.10)
    # Small static feature
    base += _make_ellipse((N, N), (cx + 18, cy - 5), (6, 6), 0, 0.20)

    # Time axis
    time_points = np.linspace(0, 60.0, T)  # 0 to 60 seconds

    # Dynamic enhancing regions (masks)
    region1 = _make_ellipse((N, N), (cx - 15, cy + 15), (8, 6), 10, 1.0)
    region2 = _make_ellipse((N, N), (cx + 12, cy - 12), (6, 8), -15, 1.0)

    # Contrast uptake curves (different timing for each region)
    curve1 = gamma_variate(time_points, A=0.6, t_arrival=5.0,
                           t_peak=12.0, alpha=3.0)
    curve2 = gamma_variate(time_points, A=0.45, t_arrival=10.0,
                           t_peak=15.0, alpha=2.5)

    # Build time series
    phantom = np.zeros((T, N, N), dtype=np.float64)
    for t_idx in range(T):
        phantom[t_idx] = base + curve1[t_idx] * region1 + curve2[t_idx] * region2

    return phantom, time_points


def generate_variable_density_mask(N, sampling_rate=0.25, center_fraction=0.08,
                                   seed=None):
    """
    Generate a 2D variable-density random point undersampling mask.

    Fully samples a central square region of k-space and randomly samples
    remaining points with probability decaying away from center (Gaussian
    variable density). This produces incoherent aliasing artifacts that
    temporal TV regularization can effectively suppress.

    Parameters
    ----------
    N : int
        k-space size (NxN).
    sampling_rate : float
        Fraction of k-space points to sample overall (~0.20-0.25).
    center_fraction : float
        Fraction of k-space (by side length) to always sample.
    seed : int or None
        Random seed.

    Returns
    -------
    mask : ndarray, (N, N) float64
        Binary mask (1 = sampled).
    """
    rng = np.random.RandomState(seed)

    mask = np.zeros((N, N), dtype=np.float64)

    # Always sample center region
    n_center = max(int(center_fraction * N), 2)
    c_start = (N - n_center) // 2
    c_end = c_start + n_center
    mask[c_start:c_end, c_start:c_end] = 1.0

    n_center_pts = int(mask.sum())
    n_total = int(sampling_rate * N * N)
    n_outer = max(n_total - n_center_pts, 0)

    # Variable-density probability map (Gaussian)
    yy, xx = np.mgrid[:N, :N]
    sigma = N / 3.0
    prob_map = np.exp(-0.5 * (((yy - N / 2) / sigma) ** 2 +
                               ((xx - N / 2) / sigma) ** 2))

    # Zero out center (already sampled)
    prob_map[c_start:c_end, c_start:c_end] = 0.0
    prob_map_flat = prob_map.ravel()

    total_prob = prob_map_flat.sum()
    if total_prob > 0:
        prob_map_flat /= total_prob
        outer_indices = rng.choice(N * N, size=min(n_outer, int((prob_map_flat > 0).sum())),
                                   replace=False, p=prob_map_flat)
        mask_flat = mask.ravel()
        mask_flat[outer_indices] = 1.0
        mask = mask_flat.reshape(N, N)

    return mask


def generate_dce_data(N=128, T=20, sampling_rate=0.25, center_fraction=0.08,
                      noise_level=0.02, seed=42):
    """
    Generate complete synthetic DCE-MRI dataset.

    Parameters
    ----------
    N : int
        Image size.
    T : int
        Number of time frames.
    sampling_rate : float
        k-space undersampling rate per frame.
    center_fraction : float
        Center k-space fraction always sampled.
    noise_level : float
        Standard deviation of complex Gaussian noise (relative to max signal).
    seed : int
        Random seed.

    Returns
    -------
    phantom : ndarray, (T, N, N)
        Ground truth dynamic images.
    kspace : ndarray, (T, N, N) complex128
        Undersampled noisy k-space per frame.
    masks : ndarray, (T, N, N)
        Undersampling masks per frame.
    time_points : ndarray, (T,)
        Time axis in seconds.
    """
    rng = np.random.RandomState(seed)
    phantom, time_points = make_dynamic_phantom(N, T)

    # Full k-space
    full_kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(phantom, axes=(-2, -1)), norm='ortho'),
        axes=(-2, -1)
    )

    # Generate different masks per frame
    masks = np.zeros((T, N, N), dtype=np.float64)
    for t in range(T):
        masks[t] = generate_variable_density_mask(
            N, sampling_rate=sampling_rate,
            center_fraction=center_fraction, seed=seed + t
        )

    # Apply masks and add noise
    noise_std = noise_level * np.abs(full_kspace).max()
    noise = noise_std * (rng.randn(T, N, N) + 1j * rng.randn(T, N, N)) / np.sqrt(2)

    kspace = full_kspace * masks + noise * masks  # noise only on sampled locations

    return phantom, kspace, masks, time_points


def save_data(output_dir, phantom, kspace, masks, time_points,
              N=128, T=20, sampling_rate=0.25, noise_level=0.02):
    """
    Save generated data in standard benchmark format.

    Parameters
    ----------
    output_dir : str
        Directory to save data files.
    phantom, kspace, masks, time_points : ndarrays
        Outputs from generate_dce_data().
    N, T, sampling_rate, noise_level : scalars
        Parameters for metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    # raw_data.npz: observations + masks (batch-first)
    np.savez_compressed(
        os.path.join(output_dir, 'raw_data.npz'),
        undersampled_kspace=kspace[np.newaxis].astype(np.complex64),  # (1, T, N, N)
        undersampling_masks=masks[np.newaxis].astype(np.float32),      # (1, T, N, N)
    )

    # ground_truth.npz (batch-first)
    np.savez_compressed(
        os.path.join(output_dir, 'ground_truth.npz'),
        dynamic_images=phantom[np.newaxis].astype(np.float32),  # (1, T, N, N)
        time_points=time_points.astype(np.float32),             # (T,)
    )

    # meta_data.json: imaging parameters only
    meta = {
        'image_size': N,
        'num_frames': T,
        'sampling_rate': sampling_rate,
        'center_fraction': 0.08,
        'noise_level': noise_level,
        'time_range_seconds': [float(time_points[0]), float(time_points[-1])],
        'modality': 'DCE-MRI',
        'description': 'Synthetic dynamic contrast-enhanced MRI with '
                       'variable-density Cartesian undersampling',
    }
    with open(os.path.join(output_dir, 'meta_data.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    print('Generating synthetic DCE-MRI data ...')
    N, T = 128, 20
    sampling_rate = 0.25
    noise_level = 0.02

    phantom, kspace, masks, time_points = generate_dce_data(
        N=N, T=T, sampling_rate=sampling_rate, noise_level=noise_level, seed=42
    )

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    save_data(data_dir, phantom, kspace, masks, time_points,
              N=N, T=T, sampling_rate=sampling_rate, noise_level=noise_level)

    print(f'  Phantom shape: {phantom.shape}')
    print(f'  k-space shape: {kspace.shape}')
    print(f'  Masks shape:   {masks.shape}')
    actual_rate = masks.mean()
    print(f'  Actual sampling rate: {actual_rate:.3f}')
    print(f'  Saved to {data_dir}/')
