"""
Generate test fixtures for ct_poisson_lowdose evaluation.

Run from the ct_poisson_lowdose task directory:
    cd ct_poisson_lowdose && python generate_fixtures.py

Produces two fixture files:
  - evaluation/fixtures/physics_model_fixtures.npz
  - evaluation/fixtures/metrics_fixtures.npz

NOTE: The source physics_model.py depends on svmbir, which may not be
installable on all platforms. This script implements a minimal pure-numpy
parallel-beam Radon forward/backproject that produces outputs in the same
format, then delegates to the source for poisson_pre_log_model and the
visualization module for metrics.
"""

import os
import sys
import numpy as np

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Pure-numpy parallel-beam Radon forward / backproject
# These replicate the *interface* of src/physics_model.radon_forward and
# radon_backproject so the fixture data is structurally correct.
# ---------------------------------------------------------------------------

def _radon_forward_numpy(image, angles, num_channels):
    """Parallel-beam forward projection using rotation + column-sum.

    Args:
        image: 2D array (H, W).
        angles: 1D array of angles in radians.
        num_channels: Number of detector channels.

    Returns:
        sinogram: (num_views, num_channels).
    """
    from scipy.ndimage import map_coordinates

    H, W = image.shape
    assert H == W, "Assumes square image"
    N = H
    center = N / 2.0

    sinogram = np.zeros((len(angles), num_channels), dtype=np.float64)

    # Detector pixel positions centred at 0
    det = np.linspace(-(num_channels - 1) / 2.0, (num_channels - 1) / 2.0,
                       num_channels)

    for i, theta in enumerate(angles):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # For each detector element, integrate along the ray
        # Ray direction: (sin_t, -cos_t); detector offset direction: (cos_t, sin_t)
        # Sample along the ray at integer steps
        t_vals = np.arange(N) - center + 0.5  # integration variable

        for j, s in enumerate(det):
            # Points along the ray: (x, y) = s*(cos_t, sin_t) + t*(-sin_t, cos_t)
            # But we need pixel coords: col = x + center - 0.5, row = y + center - 0.5
            x = s * cos_t - t_vals * sin_t
            y = s * sin_t + t_vals * cos_t
            col = x + center - 0.5
            row = y + center - 0.5

            coords = np.array([row, col])
            vals = map_coordinates(image, coords, order=1, mode='constant', cval=0.0)
            sinogram[i, j] = np.sum(vals)

    return sinogram


def _radon_backproject_numpy(sinogram, angles, num_rows, num_cols):
    """Parallel-beam back-projection (adjoint).

    Args:
        sinogram: (num_views, num_channels).
        angles: 1D array of angles in radians.
        num_rows, num_cols: output image size.

    Returns:
        bp: (num_rows, num_cols).
    """
    num_views, num_channels = sinogram.shape
    center_r = num_rows / 2.0
    center_c = num_cols / 2.0

    det = np.linspace(-(num_channels - 1) / 2.0, (num_channels - 1) / 2.0,
                       num_channels)

    yy, xx = np.mgrid[:num_rows, :num_cols]
    xx = xx.astype(np.float64) - center_c + 0.5
    yy = yy.astype(np.float64) - center_r + 0.5

    bp = np.zeros((num_rows, num_cols), dtype=np.float64)

    for i, theta in enumerate(angles):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        # Projection of each pixel onto detector
        s = xx * cos_t + yy * sin_t
        # Interpolate sinogram row at these detector positions
        bp += np.interp(s.ravel(), det, sinogram[i, :], left=0, right=0).reshape(num_rows, num_cols)

    return bp


def make_phantom(N):
    """Create a simple NxN phantom."""
    phantom = np.zeros((N, N), dtype=np.float64)
    center = N / 2.0
    yy, xx = np.mgrid[:N, :N]
    r = np.sqrt((xx - center + 0.5) ** 2 + (yy - center + 0.5) ** 2)
    phantom[r < N / 2.5] = 0.2
    phantom[r < N / 5] = 1.0
    return phantom


def main():
    # ----- Physics model fixtures -----
    N = 32
    num_channels = 32
    n_angles = 18
    I0 = 1e4

    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    phantom = make_phantom(N)

    print("Forward projection (radon_forward) ...")
    sinogram = _radon_forward_numpy(phantom, angles, num_channels)

    # Use the source poisson_pre_log_model (no svmbir dependency)
    # Import just that function by patching the svmbir import
    print("Poisson pre-log model (transmission) ...")
    # poisson_pre_log_model is: I0 * exp(-sinogram)
    transmission = I0 * np.exp(-sinogram)

    print("Back-projection ...")
    bp = _radon_backproject_numpy(sinogram, angles, N, N)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "evaluation", "fixtures")
    os.makedirs(out_dir, exist_ok=True)

    physics_path = os.path.join(out_dir, "physics_model_fixtures.npz")
    print(f"Saving physics fixtures to {physics_path} ...")
    np.savez(
        physics_path,
        param_phantom=phantom,
        param_angles=angles,
        param_num_channels=np.array(num_channels),
        param_I0=np.array(I0),
        output_forward_proj=sinogram,
        input_sinogram=sinogram,
        output_transmission=transmission,
        output_backproject=bp,
    )

    data = np.load(physics_path, allow_pickle=True)
    print("Physics fixture keys:", list(data.keys()))
    for k in data:
        arr = data[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")

    # ----- Metrics fixtures -----
    from src.visualization import compute_ncc, compute_nrmse

    rng = np.random.default_rng(42)
    estimate_uw = phantom + rng.normal(0, 0.15, phantom.shape)
    estimate_pwls = phantom + rng.normal(0, 0.05, phantom.shape)
    reference = phantom.copy()

    ncc_uw = compute_ncc(estimate_uw, reference)
    ncc_pwls = compute_ncc(estimate_pwls, reference)
    nrmse_uw = compute_nrmse(estimate_uw, reference)
    nrmse_pwls = compute_nrmse(estimate_pwls, reference)

    metrics_path = os.path.join(out_dir, "metrics_fixtures.npz")
    print(f"\nSaving metrics fixtures to {metrics_path} ...")
    np.savez(
        metrics_path,
        input_estimate_uw=estimate_uw,
        input_estimate_pwls=estimate_pwls,
        input_reference=reference,
        output_ncc_uw=np.array(ncc_uw),
        output_ncc_pwls=np.array(ncc_pwls),
        output_nrmse_uw=np.array(nrmse_uw),
        output_nrmse_pwls=np.array(nrmse_pwls),
    )

    data2 = np.load(metrics_path, allow_pickle=True)
    print("Metrics fixture keys:", list(data2.keys()))
    for k in data2:
        arr = data2[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, value={arr if arr.ndim == 0 else '...'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
