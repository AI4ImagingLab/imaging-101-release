"""
Visualization Utilities for SSNP-ODT
======================================
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_ri_slices(volume: np.ndarray, slice_indices: list = None,
                   title: str = "RI Contrast Slices",
                   vmin=None, vmax=None, figsize=None):
    """
    XY cross-sections of a 3D RI volume at selected z positions.

    Parameters
    ----------
    volume : (Nz, Ny, Nx) real array
    slice_indices : list of int, or None (auto-select 5 evenly spaced)
    """
    nz = volume.shape[0]
    if slice_indices is None:
        slice_indices = np.linspace(nz // 4, 3 * nz // 4, 5, dtype=int).tolist()

    n = len(slice_indices)
    if figsize is None:
        figsize = (3.5 * n, 3.5)
    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    if n == 1:
        axes = [axes]

    if vmax is None:
        vmax = np.percentile(volume, 99.5)
    if vmin is None:
        vmin = 0.0

    for ax, idx in zip(axes, slice_indices):
        im = ax.imshow(volume[idx], cmap="hot", origin="lower",
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"z = {idx}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(title, fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label="Δn")
    return fig


def plot_xz_cross_section(volume: np.ndarray, y_index: int = None,
                          title: str = "XZ Cross Section",
                          vmin=None, vmax=None, figsize=(8, 4)):
    """
    XZ cross-section through the RI volume.

    Parameters
    ----------
    volume : (Nz, Ny, Nx) real array
    y_index : int or None (defaults to Ny//2)
    """
    if y_index is None:
        y_index = volume.shape[1] // 2
    if vmax is None:
        vmax = np.percentile(volume, 99.5)
    if vmin is None:
        vmin = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    xz_slice = volume[:, y_index, :]
    im = ax.imshow(xz_slice, cmap="hot", origin="lower",
                   vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"{title} (y={y_index})")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label="Δn")
    fig.tight_layout()
    return fig


def plot_comparison(ground_truth: np.ndarray, reconstruction: np.ndarray,
                    slice_idx: int = None, figsize=(14, 8)):
    """
    Side-by-side comparison: ground truth vs reconstruction (XY and XZ).

    Parameters
    ----------
    ground_truth : (Nz, Ny, Nx) real array
    reconstruction : (Nz, Ny, Nx) real array
    slice_idx : int or None (defaults to Nz//2)
    """
    nz = ground_truth.shape[0]
    ny = ground_truth.shape[1]
    if slice_idx is None:
        slice_idx = nz // 2

    vmax = max(np.percentile(ground_truth, 99.5),
               np.percentile(reconstruction, 99.5))
    vmin = 0.0

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    axes[0, 0].imshow(ground_truth[slice_idx], cmap="hot", origin="lower",
                      vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"Ground Truth (z={slice_idx})")

    axes[0, 1].imshow(reconstruction[slice_idx], cmap="hot", origin="lower",
                      vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"Reconstruction (z={slice_idx})")

    diff = np.abs(ground_truth[slice_idx] - reconstruction[slice_idx])
    im_diff = axes[0, 2].imshow(diff, cmap="hot", origin="lower")
    axes[0, 2].set_title("Absolute Difference")
    fig.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)

    y_mid = ny // 2
    axes[1, 0].imshow(ground_truth[:, y_mid, :], cmap="hot", origin="lower",
                      vmin=vmin, vmax=vmax, aspect="auto")
    axes[1, 0].set_title(f"GT XZ (y={y_mid})")

    axes[1, 1].imshow(reconstruction[:, y_mid, :], cmap="hot", origin="lower",
                      vmin=vmin, vmax=vmax, aspect="auto")
    axes[1, 1].set_title(f"Recon XZ (y={y_mid})")

    diff_xz = np.abs(ground_truth[:, y_mid, :] - reconstruction[:, y_mid, :])
    im_diff2 = axes[1, 2].imshow(diff_xz, cmap="hot", origin="lower",
                                 aspect="auto")
    axes[1, 2].set_title("XZ Difference")
    fig.colorbar(im_diff2, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y" if ax == axes[0, 0] else "z")

    fig.suptitle("SSNP-ODT Reconstruction Comparison", fontsize=13)
    fig.tight_layout()
    return fig


def plot_loss_history(losses: list, title: str = "Loss History",
                      figsize=(6, 4)):
    """
    Convergence curve of the complex field MSE loss.

    Parameters
    ----------
    losses : list of float
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, len(losses) + 1), losses, "o-", color="steelblue", lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Complex Field MSE Loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_measurements(measurements: np.ndarray, n_cols: int = 4,
                      figsize=None):
    """
    Amplitude and phase panels for each illumination angle.

    Parameters
    ----------
    measurements : (n_angles, Ny, Nx) complex array
    """
    n_angles = measurements.shape[0]
    n_rows = (n_angles + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (3.5 * n_cols, 7 * n_rows)

    fig, axes = plt.subplots(2 * n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    amp = np.abs(measurements)
    phase = np.angle(measurements)

    for i in range(n_angles):
        r, c = divmod(i, n_cols)
        theta_deg = 360.0 * i / n_angles

        axes[2 * r, c].imshow(amp[i], cmap="gray", origin="lower")
        axes[2 * r, c].set_title(f"Amp {i} ({theta_deg:.0f}°)", fontsize=9)
        axes[2 * r, c].axis("off")

        axes[2 * r + 1, c].imshow(phase[i], cmap="hsv", origin="lower",
                                   vmin=-np.pi, vmax=np.pi)
        axes[2 * r + 1, c].set_title(f"Phase {i} ({theta_deg:.0f}°)", fontsize=9)
        axes[2 * r + 1, c].axis("off")

    for i in range(n_angles, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[2 * r, c].axis("off")
        axes[2 * r + 1, c].axis("off")

    fig.suptitle("Simulated ODT Measurements (Amplitude & Phase)", fontsize=12)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(estimate: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Compute reconstruction quality metrics.

    NCC: cosine similarity (no mean subtraction) per CLAUDE.md convention.
    NRMSE: RMS error normalised by dynamic range of ground truth.

    Parameters
    ----------
    estimate : (Nz, Ny, Nx) array — reconstructed volume
    ground_truth : (Nz, Ny, Nx) array — reference volume

    Returns
    -------
    dict with keys 'nrmse', 'ncc', 'ssim'
    """
    gt = ground_truth.astype(np.float64)
    est = estimate.astype(np.float64)

    nrmse = np.sqrt(np.mean((est - gt) ** 2)) / (gt.max() - gt.min() + 1e-12)

    gt_flat = gt.ravel()
    est_flat = est.ravel()
    ncc = float(np.dot(gt_flat, est_flat) / (
        np.linalg.norm(gt_flat) * np.linalg.norm(est_flat) + 1e-12))

    data_range = max(gt.max() - gt.min(), est.max() - est.min(), 1e-12)
    ssim_vals = [ssim(gt[iz], est[iz], data_range=data_range)
                 for iz in range(gt.shape[0])]
    ssim_mean = float(np.mean(ssim_vals))

    return {"nrmse": float(nrmse), "ncc": ncc, "ssim": ssim_mean}


def print_metrics_table(metrics: dict) -> None:
    """Print a formatted metrics table."""
    print(f"  {'Metric':<12} {'Value':>10}")
    print(f"  {'─'*12} {'─'*10}")
    print(f"  {'NRMSE':<12} {metrics['nrmse']:>10.4f}")
    print(f"  {'NCC':<12} {metrics['ncc']:>10.4f}")
    print(f"  {'SSIM':<12} {metrics['ssim']:>10.4f}")
