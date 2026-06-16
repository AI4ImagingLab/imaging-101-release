"""Visualization utilities for weather radar data assimilation."""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# VIL colormap matching SEVIR conventions
VIL_COLORS = [
    "#FFFFFF",  # 0 - no echo
    "#00FFFF",  # 1 - light blue
    "#0080FF",  # 2 - blue
    "#0000FF",  # 3 - dark blue
    "#00FF00",  # 4 - green
    "#00C000",  # 5 - dark green
    "#008000",  # 6 - darker green
    "#FFFF00",  # 7 - yellow
    "#E0C000",  # 8 - gold
    "#FF8000",  # 9 - orange
    "#FF0000",  # 10 - red
    "#C00000",  # 11 - dark red
    "#800000",  # 12 - maroon
    "#FF00FF",  # 13 - magenta
    "#800080",  # 14 - purple
    "#400040",  # 15 - dark purple
]

VIL_LEVELS = [
    0, 16, 31, 59, 74, 100, 133, 160, 181, 200, 219, 235, 255, 275, 300, 350, 400
]


def get_vil_colormap():
    """Return the standard VIL radar colormap and normalization."""
    cmap = ListedColormap(VIL_COLORS)
    norm = BoundaryNorm([v / 400 for v in VIL_LEVELS], cmap.N)
    return cmap, norm


def plot_comparison(
    condition,
    ground_truth,
    observations,
    reconstruction,
    save_path,
    mask=None,
):
    """Plot side-by-side comparison of condition, GT, observations, and reconstruction.

    Parameters
    ----------
    condition : np.ndarray
        Past frames, shape (n_cond, H, W), values in [0, 1].
    ground_truth : np.ndarray
        True future frames, shape (n_pred, H, W).
    observations : np.ndarray
        Sparse observations, shape (n_pred, H, W).
    reconstruction : np.ndarray
        Reconstructed frames, shape (n_pred, H, W).
    save_path : str
        Path to save the figure.
    mask : np.ndarray, optional
        Observation mask for visualization.
    """
    n_cond = condition.shape[0]
    n_pred = ground_truth.shape[0]
    n_rows = 4  # condition, observation, ground truth, reconstruction
    n_cols = max(n_cond, n_pred)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # Row 0: Condition frames
    for j in range(n_cond):
        axes[0, j].imshow(condition[j], cmap="gray", vmin=0, vmax=1)
        if j == 0:
            axes[0, j].set_ylabel("Condition", fontsize=12, rotation=0, labelpad=80, va="center")

    # Row 1: Sparse observations
    for j in range(n_pred):
        axes[1, j].imshow(observations[j], cmap="gray", vmin=0, vmax=1)
        if j == 0:
            axes[1, j].set_ylabel("Observation", fontsize=12, rotation=0, labelpad=80, va="center")

    # Row 2: Ground truth
    for j in range(n_pred):
        axes[2, j].imshow(ground_truth[j], cmap="gray", vmin=0, vmax=1)
        if j == 0:
            axes[2, j].set_ylabel("Ground Truth", fontsize=12, rotation=0, labelpad=80, va="center")

    # Row 3: Reconstruction
    for j in range(n_pred):
        axes[3, j].imshow(np.clip(reconstruction[j], 0, 1), cmap="gray", vmin=0, vmax=1)
        if j == 0:
            axes[3, j].set_ylabel("Reconstruction", fontsize=12, rotation=0, labelpad=80, va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def plot_error_map(ground_truth, reconstruction, save_path):
    """Plot per-frame absolute error maps.

    Parameters
    ----------
    ground_truth : np.ndarray
        True frames, shape (n_pred, H, W).
    reconstruction : np.ndarray
        Reconstructed frames, shape (n_pred, H, W).
    save_path : str
        Path to save the figure.
    """
    n_pred = ground_truth.shape[0]
    error = np.abs(ground_truth - np.clip(reconstruction, 0, 1))

    fig, axes = plt.subplots(1, n_pred, figsize=(4 * n_pred, 4))
    if n_pred == 1:
        axes = [axes]

    for j in range(n_pred):
        im = axes[j].imshow(error[j], cmap="hot", vmin=0, vmax=0.3)
        axes[j].set_title(f"Frame {j+1}")
        axes[j].axis("off")
        plt.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)

    plt.suptitle("Absolute Error", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved error map to {save_path}")
