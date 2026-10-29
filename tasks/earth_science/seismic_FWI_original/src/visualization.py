"""Visualization utilities for seismic FWI."""

import numpy as np
import matplotlib.pyplot as plt


def plot_velocity_models(
    v_true: np.ndarray,
    v_init: np.ndarray,
    v_inv: np.ndarray,
    dx: float,
    dz: float,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot three velocity models (initial, inverted, true) as colormaps.

    Args:
        v_true: True velocity, shape (ny, nx), in m/s.
        v_init: Initial velocity, same shape.
        v_inv: Inverted velocity, same shape.
        dx: Horizontal grid spacing in meters.
        dz: Vertical grid spacing in meters.
        save_path: If given, save figure to this path.

    Returns:
        fig: Matplotlib figure.
    """
    ny, nx = v_true.shape
    extent = [0, (ny - 1) * dx, (nx - 1) * dz, 0]
    vmin, vmax = v_true.min(), v_true.max()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    titles = ["Initial Velocity Model", "Inverted Velocity Model", "True Velocity Model"]
    models = [v_init, v_inv, v_true]

    for ax, model, title in zip(axes, models, titles):
        im = ax.imshow(
            model.T, cmap="viridis", aspect="auto",
            vmin=vmin, vmax=vmax, extent=extent
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        fig.colorbar(im, ax=ax, label="Velocity (m/s)", shrink=0.9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shot_gather(
    data: np.ndarray,
    shot_idx: int = 0,
    title: str = "Shot gather",
    clip_percentile: float = 0.95,
) -> plt.Figure:
    """
    Plot a seismic shot gather (receivers × time) as a grayscale image.

    Args:
        data: Seismic data array, shape (n_shots, n_receivers, nt).
        shot_idx: Shot index to display. Default 0.
        title: Plot title.
        clip_percentile: Clip amplitude at this quantile for display. Default 0.95.

    Returns:
        fig: Matplotlib figure.
    """
    trace = data[shot_idx]  # (n_receivers, nt)
    amp = np.quantile(np.abs(trace), clip_percentile)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        trace.T, cmap="gray", aspect="auto",
        vmin=-amp, vmax=amp
    )
    ax.set_title(f"{title} (shot {shot_idx + 1})", fontsize=13)
    ax.set_xlabel("Receiver index")
    ax.set_ylabel("Time sample")
    fig.colorbar(ax.images[0], ax=ax, label="Amplitude")
    plt.tight_layout()
    return fig


def plot_loss_curve(losses: list) -> plt.Figure:
    """
    Plot FWI loss (MSE misfit) vs. iteration number on a logarithmic scale.

    Args:
        losses: List of loss values, one per epoch.

    Returns:
        fig: Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(losses)
    ax.set_title("FWI Loss Convergence", fontsize=13)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_data_comparison(
    obs: np.ndarray,
    pred: np.ndarray,
    shot_idx: int = 0,
) -> plt.Figure:
    """
    Three-panel plot: observed | predicted | residual for one shot.

    Args:
        obs: Observed seismograms, shape (n_shots, n_receivers, nt).
        pred: Predicted seismograms, same shape.
        shot_idx: Shot index to compare. Default 0.

    Returns:
        fig: Matplotlib figure.
    """
    o = obs[shot_idx]    # (n_receivers, nt)
    p = pred[shot_idx]
    r = p - o

    amp = np.quantile(np.abs(o), 0.95)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(o.T, cmap="gray", aspect="auto", vmin=-amp, vmax=amp)
    axes[0].set_title(f"Observed (shot {shot_idx+1})", fontsize=13)
    axes[0].set_xlabel("Receiver"); axes[0].set_ylabel("Time sample")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(p.T, cmap="gray", aspect="auto", vmin=-amp, vmax=amp)
    axes[1].set_title(f"Predicted (shot {shot_idx+1})", fontsize=13)
    axes[1].set_xlabel("Receiver")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(r.T, cmap="gray", aspect="auto")
    axes[2].set_title(f"Residual (pred − obs) (shot {shot_idx+1})", fontsize=13)
    axes[2].set_xlabel("Receiver")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    return fig


def compute_data_metrics(
    obs: np.ndarray,
    pred: np.ndarray,
    shot_idx: int = 0,
) -> dict:
    """
    Compute MSE, RMS error, and relative L2 error for one shot gather.

    Args:
        obs: Observed data, shape (n_shots, n_receivers, nt).
        pred: Predicted data, same shape.
        shot_idx: Shot index. Default 0.

    Returns:
        metrics: dict with keys 'mse', 'rms', 'rel_l2'.
    """
    o = obs[shot_idx].astype(np.float64)
    p = pred[shot_idx].astype(np.float64)
    res = p - o
    mse = float(np.mean(res ** 2))
    rms = float(np.sqrt(mse))
    rel_l2 = float(np.linalg.norm(res) / (np.linalg.norm(o) + 1e-12))
    return {"mse": mse, "rms": rms, "rel_l2": rel_l2}
