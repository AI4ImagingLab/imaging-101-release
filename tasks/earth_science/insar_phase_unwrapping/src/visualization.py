"""Visualization utilities for InSAR phase unwrapping results."""

import numpy as np
import matplotlib.pyplot as plt


def plot_wrapped_phase_and_coherence(wrapped_phase, coherence, save_path=None):
    """Plot wrapped interferogram phase and coherence side by side (Paper Fig. 1).

    Parameters
    ----------
    wrapped_phase : ndarray
    coherence : ndarray
    save_path : str, optional
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(wrapped_phase, cmap="RdBu_r", vmin=-np.pi, vmax=np.pi)
    axes[0].set_title("Wrapped Interferogram Phase")
    axes[0].set_xlabel("Range")
    axes[0].set_ylabel("Azimuth")
    plt.colorbar(im0, ax=axes[0], label="Phase (rad)")

    im1 = axes[1].imshow(coherence, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Interferogram Coherence")
    axes[1].set_xlabel("Range")
    axes[1].set_ylabel("Azimuth")
    plt.colorbar(im1, ax=axes[1], label="Coherence")

    plt.suptitle("Fig. 1: Wrapped Phase and Coherence", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_unwrapped_comparison(unwrapped_spurs, unwrapped_snaphu, save_path=None):
    """Plot SPURS vs SNAPHU unwrapped phase side by side (Paper Fig. 2).

    Aligns the constant offset before plotting.

    Parameters
    ----------
    unwrapped_spurs : ndarray
    unwrapped_snaphu : ndarray
    save_path : str, optional
    """
    # Align constant offset
    offset = unwrapped_snaphu.mean() - unwrapped_spurs.mean()
    spurs_aligned = unwrapped_spurs + offset

    vmin = min(spurs_aligned.min(), unwrapped_snaphu.min())
    vmax = max(spurs_aligned.max(), unwrapped_snaphu.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(spurs_aligned, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("Unwrapped Phase (SPURS)")
    axes[0].set_xlabel("Range")
    axes[0].set_ylabel("Azimuth")
    plt.colorbar(im0, ax=axes[0], label="Phase (rad)")

    im1 = axes[1].imshow(unwrapped_snaphu, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("Unwrapped Phase (SNAPHU)")
    axes[1].set_xlabel("Range")
    axes[1].set_ylabel("Azimuth")
    plt.colorbar(im1, ax=axes[1], label="Phase (rad)")

    plt.suptitle("Fig. 2: SPURS vs SNAPHU Unwrapped Phase", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_residuals(unwrapped_spurs, unwrapped_snaphu, wrapped_phase, save_path=None):
    """Plot residuals (unwrapped - wrapped) for both methods (Paper Fig. 3).

    Parameters
    ----------
    unwrapped_spurs : ndarray
    unwrapped_snaphu : ndarray
    wrapped_phase : ndarray
    save_path : str, optional
    """
    # Align constant offset
    offset = unwrapped_snaphu.mean() - unwrapped_spurs.mean()
    spurs_aligned = unwrapped_spurs + offset

    diff_spurs = spurs_aligned - wrapped_phase
    diff_snaphu = unwrapped_snaphu - wrapped_phase

    p2, p98 = np.percentile(
        np.concatenate([diff_spurs.ravel(), diff_snaphu.ravel()]), [2, 98])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(diff_spurs, cmap="RdBu_r", vmin=p2, vmax=p98)
    axes[0].set_title("Unwrapped - Wrapped (SPURS)")
    axes[0].set_xlabel("Range")
    axes[0].set_ylabel("Azimuth")
    plt.colorbar(im0, ax=axes[0], label="Phase diff (rad)")

    im1 = axes[1].imshow(diff_snaphu, cmap="RdBu_r", vmin=p2, vmax=p98)
    axes[1].set_title("Unwrapped - Wrapped (SNAPHU)")
    axes[1].set_xlabel("Range")
    axes[1].set_ylabel("Azimuth")
    plt.colorbar(im1, ax=axes[1], label="Phase diff (rad)")

    plt.suptitle("Fig. 3: Residuals (Unwrapped - Wrapped)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_difference_map(unwrapped_spurs, unwrapped_snaphu, save_path=None):
    """Plot direct pixel-wise difference between SPURS and SNAPHU.

    Parameters
    ----------
    unwrapped_spurs : ndarray
    unwrapped_snaphu : ndarray
    save_path : str, optional
    """
    spurs_c = unwrapped_spurs - unwrapped_spurs.mean()
    snaphu_c = unwrapped_snaphu - unwrapped_snaphu.mean()
    diff = spurs_c - snaphu_c

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-np.pi, vmax=np.pi)
    ax.set_title("SPURS - SNAPHU (mean-removed)")
    ax.set_xlabel("Range")
    ax.set_ylabel("Azimuth")
    plt.colorbar(im, ax=ax, label="Phase difference (rad)")
    plt.suptitle("Direct Comparison: SPURS vs SNAPHU", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compute_metrics(unwrapped_spurs, unwrapped_snaphu):
    """Compute comparison metrics between SPURS and SNAPHU results.

    Parameters
    ----------
    unwrapped_spurs : ndarray
    unwrapped_snaphu : ndarray

    Returns
    -------
    dict
    """
    spurs_c = unwrapped_spurs - unwrapped_spurs.mean()
    snaphu_c = unwrapped_snaphu - unwrapped_snaphu.mean()
    diff = spurs_c - snaphu_c

    return {
        "mean_abs_diff_rad": float(np.abs(diff).mean()),
        "max_abs_diff_rad": float(np.abs(diff).max()),
        "rmse_rad": float(np.sqrt(np.mean(diff ** 2))),
        "fraction_within_pi": float((np.abs(diff) < np.pi).mean()),
        "fraction_within_2pi": float((np.abs(diff) < 2 * np.pi).mean()),
        "n_pixels_disagree_gt_pi": int((np.abs(diff) > np.pi).sum()),
        "total_pixels": int(diff.size),
    }
