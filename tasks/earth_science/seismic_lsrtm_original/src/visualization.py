"""Visualization utilities for seismic LSRTM."""

import numpy as np
import matplotlib.pyplot as plt


def plot_scatter_image(
    scatter: np.ndarray,
    dx: float,
    title: str = "LSRTM Image",
    percentile_clip: float = 95.0,
) -> plt.Figure:
    """Plot the scattering potential (reflectivity image).

    Args:
        scatter: Scattering potential, shape (ny, nx).
        dx: Grid spacing in meters.
        title: Plot title.
        percentile_clip: Clip at this percentile for display. Default 95.

    Returns:
        fig: Matplotlib figure.
    """
    ny, nx = scatter.shape
    extent = [0, (ny - 1) * dx, (nx - 1) * dx, 0]
    vmin, vmax = np.percentile(scatter, [100 - percentile_clip, percentile_clip])

    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    im = ax.imshow(scatter.T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax, extent=extent)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    fig.colorbar(im, ax=ax, label="Scattering potential", shrink=0.9)
    plt.tight_layout()
    return fig


def plot_velocity_models(
    v_true: np.ndarray,
    v_mig: np.ndarray,
    dx: float,
) -> plt.Figure:
    """Plot true and migration velocity models.

    Args:
        v_true: True velocity, shape (ny, nx), in m/s.
        v_mig: Migration velocity, same shape.
        dx: Grid spacing in meters.

    Returns:
        fig: Matplotlib figure.
    """
    ny, nx = v_true.shape
    extent = [0, (ny - 1) * dx, (nx - 1) * dx, 0]
    vmin, vmax = v_true.min(), v_true.max()

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6), sharex=True)

    im0 = axes[0].imshow(v_true.T, aspect="auto", cmap="viridis", extent=extent, vmin=vmin, vmax=vmax)
    axes[0].set_title("True Velocity Model")
    axes[0].set_ylabel("Depth (m)")
    fig.colorbar(im0, ax=axes[0], label="Velocity (m/s)", shrink=0.8)

    im1 = axes[1].imshow(v_mig.T, aspect="auto", cmap="viridis", extent=extent, vmin=vmin, vmax=vmax)
    axes[1].set_title("Migration Velocity")
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Depth (m)")
    fig.colorbar(im1, ax=axes[1], label="Velocity (m/s)", shrink=0.8)

    plt.tight_layout()
    return fig


def plot_scattered_data(
    observed: np.ndarray,
    direct: np.ndarray,
    scattered: np.ndarray,
    shot_idx: int = 0,
) -> plt.Figure:
    """Three-panel plot: observed | direct | scattered for one shot.

    Args:
        observed: Full observed data, shape (n_shots, n_rec, nt).
        direct: Direct arrivals, same shape.
        scattered: Scattered data (observed - direct), same shape.
        shot_idx: Shot index. Default 0.

    Returns:
        fig: Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True, sharey=True)

    axes[0].imshow(observed[shot_idx].T, aspect="auto", cmap="gray")
    axes[0].set_title("Observed")

    axes[1].imshow(direct[shot_idx].T, aspect="auto", cmap="gray")
    axes[1].set_title("Direct (predicted)")

    axes[2].imshow(scattered[shot_idx].T, aspect="auto", cmap="gray")
    axes[2].set_title("Scattered (obs - direct)")

    axes[0].set_ylabel("Time sample")
    for ax in axes:
        ax.set_xlabel("Receiver")

    plt.tight_layout()
    return fig


def plot_data_comparison(
    obs: np.ndarray,
    pred: np.ndarray,
    shot_idx: int = 0,
) -> plt.Figure:
    """Three-panel: observed scattered | predicted scattered | residual.

    Args:
        obs: Observed scattered data, shape (n_shots, n_rec, nt).
        pred: Predicted scattered data, same shape.
        shot_idx: Shot index. Default 0.

    Returns:
        fig: Matplotlib figure.
    """
    o, p = obs[shot_idx], pred[shot_idx]
    r = p - o
    amp = np.quantile(np.abs(o), 0.95)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, data, title in zip(axes, [o, p, r],
            [f"Obs scattered (shot {shot_idx+1})", f"Pred scattered", "Residual"]):
        ax.imshow(data.T, cmap="gray", aspect="auto", vmin=-amp, vmax=amp)
        ax.set_title(title)
        ax.set_xlabel("Receiver")
    axes[0].set_ylabel("Time sample")
    plt.tight_layout()
    return fig


def plot_loss_curve(losses: list) -> plt.Figure:
    """Plot LSRTM loss vs iteration.

    Args:
        losses: List of loss values.

    Returns:
        fig: Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, len(losses) + 1), losses, "o-")
    ax.set_title("LSRTM Loss Convergence")
    ax.set_xlabel("L-BFGS Iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def compute_data_metrics(
    obs: np.ndarray,
    pred: np.ndarray,
    shot_idx: int = 0,
) -> dict:
    """Compute MSE, RMS error, and relative L2 error for one shot gather.

    Args:
        obs: Observed data, shape (n_shots, n_rec, nt).
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
