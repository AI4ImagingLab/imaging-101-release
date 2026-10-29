"""
Visualization Utilities for Reflection-Mode ODT
================================================

Plotting functions and quantitative metrics for evaluating
reconstruction quality.
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
    Plot XY cross-sections of a 3D volume at selected z positions.

    Parameters
    ----------
    volume : (Nz, Ny, Nx) array
    slice_indices : list of int, or None (defaults to all slices for small Nz)
    """
    nz = volume.shape[0]
    if slice_indices is None:
        if nz <= 6:
            slice_indices = list(range(nz))
        else:
            slice_indices = np.linspace(0, nz - 1, 4, dtype=int).tolist()

    n = len(slice_indices)
    if figsize is None:
        figsize = (3.5 * n, 3.5)
    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    if n == 1:
        axes = [axes]

    if vmax is None:
        vmax = np.percentile(np.abs(volume), 99.5)
    if vmin is None:
        vmin = -vmax if volume.min() < 0 else 0.0

    # Use diverging colormap for negative RI contrast
    cmap = "RdBu_r" if volume.min() < 0 else "hot"

    for ax, idx in zip(axes, slice_indices):
        im = ax.imshow(volume[idx], cmap=cmap, origin="lower",
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"Layer {idx}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(title, fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label="Δn")
    return fig


def plot_comparison(ground_truth: np.ndarray, reconstruction: np.ndarray,
                    slice_idx: int = None, figsize=(14, 8)):
    """
    Side-by-side comparison: ground truth vs reconstruction.

    Shows XY slices for all layers (for small Nz) or selected layers.

    Parameters
    ----------
    ground_truth : (Nz, Ny, Nx) array
    reconstruction : (Nz, Ny, Nx) array
    slice_idx : int or None (defaults to middle slice if Nz > 4)
    """
    nz = ground_truth.shape[0]

    # For small Nz (e.g., 4 layers), show all layers
    if nz <= 4:
        fig, axes = plt.subplots(3, nz, figsize=(3.5 * nz, 10))
        slice_indices = list(range(nz))
    else:
        # For large Nz, show selected slices
        if slice_idx is None:
            slice_idx = nz // 2
        slice_indices = [slice_idx]
        fig, axes = plt.subplots(3, 1, figsize=(6, 14))
        axes = axes.reshape(3, 1)

    vmax = max(np.percentile(np.abs(ground_truth), 99.5),
               np.percentile(np.abs(reconstruction), 99.5))
    vmin = -vmax if ground_truth.min() < 0 else 0.0
    cmap = "RdBu_r" if ground_truth.min() < 0 else "hot"

    for col_idx, z_idx in enumerate(slice_indices):
        # Row 0: Ground truth
        axes[0, col_idx].imshow(ground_truth[z_idx], cmap=cmap, origin="lower",
                                vmin=vmin, vmax=vmax)
        axes[0, col_idx].set_title(f"Ground Truth (z={z_idx})")
        axes[0, col_idx].axis("off")

        # Row 1: Reconstruction
        axes[1, col_idx].imshow(reconstruction[z_idx], cmap=cmap, origin="lower",
                                vmin=vmin, vmax=vmax)
        axes[1, col_idx].set_title(f"Reconstruction (z={z_idx})")
        axes[1, col_idx].axis("off")

        # Row 2: Absolute difference
        diff = np.abs(ground_truth[z_idx] - reconstruction[z_idx])
        im_diff = axes[2, col_idx].imshow(diff, cmap="hot", origin="lower")
        axes[2, col_idx].set_title(f"|Difference| (z={z_idx})")
        axes[2, col_idx].axis("off")
        fig.colorbar(im_diff, ax=axes[2, col_idx], fraction=0.046, pad=0.04)

    fig.suptitle("Reflection-Mode ODT Reconstruction Comparison", fontsize=13)
    return fig


def plot_loss_history(losses: list, title: str = "Loss History",
                      figsize=(6, 4)):
    """
    Plot the convergence of the reconstruction loss.

    Parameters
    ----------
    losses : list of float
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, len(losses) + 1), losses, "o-", color="steelblue", lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Amplitude MSE Loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_measurements(measurements: np.ndarray, n_cols: int = 8,
                      illumination_rings: list = None,
                      figsize=None):
    """
    Plot the simulated intensity measurements for each illumination angle.

    Parameters
    ----------
    measurements : (n_angles, Ny, Nx) array
    n_cols : int
        Number of columns in the grid.
    illumination_rings : list of dict, optional
        Each entry: {"NA": float, "n_angles": int, "type": "BF" or "DF"}.
        If provided, each image is labelled with its ring type and NA.
    """
    n_angles = measurements.shape[0]
    n_rows = (n_angles + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (2.5 * n_cols, 2.5 * n_rows)

    # Build per-angle labels
    if illumination_rings is not None:
        labels = []
        for ring in illumination_rings:
            for m in range(ring["n_angles"]):
                labels.append(f"{ring['type']} NA={ring['NA']:.3f}\nAngle {m}")
    else:
        labels = [f"Angle {i}" for i in range(n_angles)]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for i in range(n_angles):
        r, c = divmod(i, n_cols)
        axes[r, c].imshow(measurements[i], cmap="gray", origin="lower")
        axes[r, c].set_title(labels[i], fontsize=7)
        axes[r, c].axis("off")

    # Hide unused axes
    for i in range(n_angles, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].axis("off")

    fig.suptitle("Simulated Reflection-Mode IDT Measurements", fontsize=12)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(estimate: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Compute reconstruction quality metrics.

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

    # NRMSE (RMS error normalised by dynamic range of ground truth)
    nrmse = np.sqrt(np.mean((est - gt)**2)) / (gt.max() - gt.min() + 1e-12)

    # Normalised cross-correlation
    gt_centered = gt - gt.mean()
    est_centered = est - est.mean()
    ncc = np.sum(gt_centered * est_centered) / (
        np.linalg.norm(gt_centered) * np.linalg.norm(est_centered) + 1e-12
    )

    # SSIM (computed slice-by-slice and averaged)
    data_range = max(gt.max() - gt.min(), est.max() - est.min(), 1e-12)
    ssim_vals = []
    for iz in range(gt.shape[0]):
        s = ssim(gt[iz], est[iz], data_range=data_range)
        ssim_vals.append(s)
    ssim_mean = float(np.mean(ssim_vals))

    return {
        "nrmse": float(nrmse),
        "ncc": float(ncc),
        "ssim": float(ssim_mean),
    }


def plot_illumination_angles(illumination_rings: list, NA_obj: float,
                             figsize=(6, 6)) -> "plt.Figure":
    """
    Visualise the illumination angle distribution in the pupil plane.

    Each illumination angle is drawn as a dot at position
    (NA·cos θ, NA·sin θ).  The objective pupil boundary is shown as a
    dashed circle at NA_obj.  BF angles (NA < NA_obj) fall inside the
    pupil; DF angles (NA > NA_obj) fall outside.

    Parameters
    ----------
    illumination_rings : list of dict
        Each entry: {"NA": float, "n_angles": int, "type": "BF" or "DF"}.
    NA_obj : float
        Objective numerical aperture (pupil boundary).
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color palette: BF shades of blue, DF shades of red
    bf_rings = [r for r in illumination_rings if r["type"] == "BF"]
    df_rings = [r for r in illumination_rings if r["type"] == "DF"]

    bf_colors = plt.cm.Blues(np.linspace(0.45, 0.85, max(len(bf_rings), 1)))
    df_colors = plt.cm.Reds(np.linspace(0.45, 0.85, max(len(df_rings), 1)))

    color_map = {}
    for i, r in enumerate(bf_rings):
        color_map[id(r)] = bf_colors[i]
    for i, r in enumerate(df_rings):
        color_map[id(r)] = df_colors[i]

    for ring in illumination_rings:
        na = ring["NA"]
        n_ang = ring["n_angles"]
        angles = 2.0 * np.pi * np.arange(n_ang) / n_ang
        kx = na * np.cos(angles)
        ky = na * np.sin(angles)
        color = color_map[id(ring)]
        label = f"{ring['type']}  NA={na:.3f}  ({n_ang} angles)"
        ax.scatter(kx, ky, s=60, color=color, label=label, zorder=3)

    # Objective pupil boundary
    theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(NA_obj * np.cos(theta), NA_obj * np.sin(theta),
            "k--", lw=1.5, label=f"Objective pupil  NA={NA_obj}")

    # Shade BF region
    from matplotlib.patches import Circle
    bf_patch = Circle((0, 0), NA_obj, color="steelblue", alpha=0.07, zorder=0)
    ax.add_patch(bf_patch)

    # Annotations
    ax.text(0, NA_obj * 0.45, "BF\nregion", ha="center", va="center",
            fontsize=9, color="steelblue", alpha=0.7)
    ax.text(0, (NA_obj + max(r["NA"] for r in illumination_rings)) / 2,
            "DF\nregion", ha="center", va="center",
            fontsize=9, color="firebrick", alpha=0.7)

    na_max = max(r["NA"] for r in illumination_rings) * 1.15
    ax.set_xlim(-na_max, na_max)
    ax.set_ylim(-na_max, na_max)
    ax.set_aspect("equal")
    ax.set_xlabel("NA$_x$ = sin θ$_x$", fontsize=11)
    ax.set_ylabel("NA$_y$ = sin θ$_y$", fontsize=11)
    ax.set_title("Illumination Angle Distribution (Pupil Plane)", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def print_metrics_table(metrics: dict) -> None:
    """
    Print a formatted metrics table.

    Parameters
    ----------
    metrics : dict with keys 'nrmse', 'ncc', 'ssim'
    """
    print(f"  {'Metric':<12} {'Value':>10}")
    print(f"  {'─'*12} {'─'*10}")
    print(f"  {'NRMSE':<12} {metrics['nrmse']:>10.4f}")
    print(f"  {'NCC':<12} {metrics['ncc']:>10.4f}")
    print(f"  {'SSIM':<12} {metrics['ssim']:>10.4f}")
