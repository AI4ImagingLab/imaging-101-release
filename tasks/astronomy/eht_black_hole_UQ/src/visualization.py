"""
Visualization for DPI — Posterior Plots and Quality Metrics
============================================================

Plotting utilities for DPI posterior mean, standard deviation, individual
samples, training loss curves, and reconstruction quality metrics.

Reference
---------
Sun & Bouman (2020), arXiv:2010.14462 — Figures 5, 6, 7
"""

import numpy as np
import matplotlib.pyplot as plt


# ── Quality Metrics ─────────────────────────────────────────────────────────

def compute_metrics(estimate: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Compute reconstruction quality metrics.

    Parameters
    ----------
    estimate : (N, N) ndarray — reconstructed (posterior mean) image
    ground_truth : (N, N) ndarray — ground-truth image

    Returns
    -------
    dict with keys:
        'nrmse'         : float — normalized root mean squared error
        'ncc'           : float — normalized cross-correlation
        'dynamic_range' : float — dynamic range in dB
    """
    # Normalize both images to unit peak
    e = estimate / (estimate.max() + 1e-30)
    g = ground_truth / (ground_truth.max() + 1e-30)

    rmse = np.sqrt(np.mean((e - g) ** 2))
    nrmse = rmse / (g.max() - g.min() + 1e-30)

    e_centered = e - e.mean()
    g_centered = g - g.mean()
    ncc = float(np.sum(e_centered * g_centered) /
                (np.sqrt(np.sum(e_centered ** 2) * np.sum(g_centered ** 2)) + 1e-30))

    peak = estimate.max()
    noise_floor = np.sqrt(np.mean((estimate - ground_truth) ** 2)) + 1e-30
    dynamic_range = float(20 * np.log10(peak / noise_floor))

    return {
        "nrmse": float(nrmse),
        "ncc": float(ncc),
        "dynamic_range": float(dynamic_range),
    }


def compute_uq_metrics(posterior_mean: np.ndarray, posterior_std: np.ndarray,
                        ground_truth: np.ndarray) -> dict:
    """
    Compute uncertainty quantification metrics.

    Parameters
    ----------
    posterior_mean : (N, N) ndarray
    posterior_std : (N, N) ndarray
    ground_truth : (N, N) ndarray

    Returns
    -------
    dict with keys:
        'calibration'      : float — fraction of GT pixels within 1-sigma of mean
        'mean_uncertainty'  : float — average posterior std
        'nrmse'            : float
        'ncc'              : float
    """
    base_metrics = compute_metrics(posterior_mean, ground_truth)

    # Calibration: fraction within 1-sigma
    within_1sigma = np.abs(ground_truth - posterior_mean) <= posterior_std
    calibration = float(np.mean(within_1sigma))

    return {
        **base_metrics,
        "calibration": calibration,
        "mean_uncertainty": float(np.mean(posterior_std)),
    }


def print_metrics_table(metrics: dict) -> None:
    """Pretty-print a metrics dictionary."""
    print(f"  {'Metric':<20s} {'Value':>10s}")
    print(f"  {'-' * 32}")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<20s} {val:>10.4f}")
        else:
            print(f"  {key:<20s} {str(val):>10s}")


# ── Plotting Functions ──────────────────────────────────────────────────────

def plot_image(image, ax=None, title=None, pixel_size_uas=None, cmap='afmhot',
               vmin=None, vmax=None):
    """Plot a single image."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))

    N = image.shape[0]
    if pixel_size_uas is not None:
        extent_uas = N * pixel_size_uas / 2
        extent = [extent_uas, -extent_uas, -extent_uas, extent_uas]
        ax.set_xlabel(r"Relative RA ($\mu$as)")
        ax.set_ylabel(r"Relative Dec ($\mu$as)")
    else:
        extent = None

    ax.imshow(image, origin='lower', cmap=cmap, extent=extent,
              vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title)
    return ax


def plot_posterior_summary(mean, std, samples, ground_truth=None,
                            pixel_size_uas=None, save_path=None):
    """
    Multi-panel posterior summary: GT | Mean | Std | Samples.

    Parameters
    ----------
    mean : (N, N) ndarray — posterior mean
    std : (N, N) ndarray — posterior standard deviation
    samples : (K, N, N) ndarray — posterior samples (first 4 shown)
    ground_truth : (N, N) ndarray or None
    pixel_size_uas : float or None
    save_path : str or None — if provided, save figure
    """
    n_samples_show = min(4, samples.shape[0])
    n_cols = 2 + (1 if ground_truth is not None else 0) + n_samples_show
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))

    col = 0
    vmax = mean.max()

    if ground_truth is not None:
        plot_image(ground_truth, axes[col], "Ground Truth",
                   pixel_size_uas, vmax=vmax)
        col += 1

    plot_image(mean, axes[col], "Posterior Mean", pixel_size_uas, vmax=vmax)
    col += 1
    plot_image(std, axes[col], "Posterior Std", pixel_size_uas, cmap='viridis')
    col += 1

    for i in range(n_samples_show):
        plot_image(samples[i], axes[col], f"Sample {i + 1}",
                   pixel_size_uas, vmax=vmax)
        col += 1

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_posterior_samples(samples, n_show=8, pixel_size_uas=None,
                            save_path=None):
    """
    Grid of posterior samples.

    Parameters
    ----------
    samples : (K, N, N) ndarray
    n_show : int — number of samples to display
    pixel_size_uas : float or None
    save_path : str or None
    """
    n_show = min(n_show, samples.shape[0])
    n_rows = int(np.ceil(n_show / 4))
    n_cols = min(4, n_show)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_show == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    vmax = np.percentile(samples[:n_show], 99)
    for i in range(n_show):
        plot_image(samples[i], axes[i], f"Sample {i + 1}",
                   pixel_size_uas, vmax=vmax)
    for i in range(n_show, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_loss_curves(loss_history, save_path=None):
    """
    Plot training loss components over epochs.

    Parameters
    ----------
    loss_history : dict
        Keys: 'total', 'cphase', 'logca', 'logdet', 'mem', 'tsv', etc.
    save_path : str or None
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    epochs = np.arange(len(loss_history['total']))

    axes[0, 0].plot(epochs, loss_history['total'])
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_yscale('symlog')

    axes[0, 1].plot(epochs, loss_history['cphase'], label='Closure Phase')
    axes[0, 1].plot(epochs, loss_history['logca'], label='Log Closure Amp')
    axes[0, 1].set_title("Data Fidelity")
    axes[0, 1].legend()

    axes[0, 2].plot(epochs, loss_history['logdet'])
    axes[0, 2].set_title("-logdet / N²")

    axes[1, 0].plot(epochs, loss_history['mem'], label='MEM')
    axes[1, 0].plot(epochs, loss_history['tsv'], label='TSV')
    axes[1, 0].plot(epochs, loss_history['l1'], label='L1')
    axes[1, 0].set_title("Image Priors")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, loss_history['flux'])
    axes[1, 1].set_title("Flux Constraint")

    axes[1, 2].plot(epochs, loss_history['center'])
    axes[1, 2].set_title("Centering Constraint")

    for ax in axes.flatten():
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
