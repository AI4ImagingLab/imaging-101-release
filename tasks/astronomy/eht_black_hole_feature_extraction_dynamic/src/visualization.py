"""
Visualization for Dynamic α-DPI Feature Extraction
====================================================

Ridge plots, parameter evolution plots, and posterior image montages
for analyzing geometric parameter distributions over time.

Reference
---------
EHT Collaboration (2022), ApJL 930:L15 — Sgr A* Paper IV, Figure 13
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ── Ground truth key mapping ────────────────────────────────────────────────
# Maps column index to ground truth dict key for SimpleCrescent (4 params).
GT_KEYS = ['diameter_uas', 'width_uas', 'asymmetry', 'position_angle_deg']


def plot_ridge(params_per_frame, param_names, gt_per_frame, weights_per_frame,
               frame_times, save_path=None):
    """
    Ridge plot of posterior parameter distributions over time.

    Each row is a time snapshot; the x-axis shows the parameter value;
    filled KDE curves show the importance-weighted posterior density.
    A red dashed line connects the ground truth values across frames.

    Similar to EHT Sgr A* Paper IV (2022) Figure 13.

    Parameters
    ----------
    params_per_frame : list of (N, D) ndarrays
        Posterior parameter samples for each frame.
    param_names : list of str
        Parameter display names.
    gt_per_frame : list of dict
        Ground truth parameter values per frame.
    weights_per_frame : list of (N,) ndarrays
        Importance weights for each frame.
    frame_times : array-like
        Time of each frame in hours.
    save_path : str or None
        If given, save figure to this path.

    Returns
    -------
    matplotlib Figure
    """
    n_params = len(param_names)
    n_frames = len(params_per_frame)
    frame_times = np.asarray(frame_times)

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 0.6 * n_frames + 2),
                             sharey=True)
    if n_params == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_frames))
    dt = frame_times[1] - frame_times[0] if n_frames > 1 else 1.0

    for j, (name, ax) in enumerate(zip(param_names, axes)):
        # Global range for this parameter
        all_vals = np.concatenate([p[:, j] for p in params_per_frame])
        gt_vals = [gt[GT_KEYS[j]] for gt in gt_per_frame]
        x_lo = min(np.percentile(all_vals, 1), min(gt_vals))
        x_hi = max(np.percentile(all_vals, 99), max(gt_vals))
        margin = (x_hi - x_lo) * 0.15
        x_range = np.linspace(x_lo - margin, x_hi + margin, 300)

        for i in range(n_frames):
            samples = params_per_frame[i][:, j]
            w = weights_per_frame[i]
            w = w / w.sum()

            try:
                kde = gaussian_kde(samples, weights=w)
                density = kde(x_range)
            except Exception:
                density = np.zeros_like(x_range)

            y_offset = frame_times[i]
            scale = dt * 0.8 / (density.max() + 1e-10)

            ax.fill_between(x_range, y_offset, y_offset + density * scale,
                            alpha=0.4, color=colors[i], zorder=2)
            ax.plot(x_range, y_offset + density * scale,
                    color=colors[i], lw=0.8, zorder=3)

        # Ground truth line
        ax.plot(gt_vals, frame_times, 'r--', lw=1.5, zorder=10, label='Truth')
        ax.scatter(gt_vals, frame_times, c='red', s=20, zorder=11, edgecolors='darkred')

        ax.set_xlabel(name, fontsize=11)
        if j == 0:
            ax.set_ylabel('Time (hr)', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_ylim(frame_times[0] - dt * 0.5, frame_times[-1] + dt * 1.2)

    axes[-1].legend(loc='upper right', fontsize=8)
    fig.suptitle('Dynamic Feature Extraction: Ridge Plot', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_param_evolution(params_per_frame, param_names, gt_per_frame,
                          weights_per_frame, frame_times, save_path=None):
    """
    Parameter evolution plot with importance-weighted mean +/- 1 sigma.

    Parameters
    ----------
    params_per_frame : list of (N, D) ndarrays
    param_names : list of str
    gt_per_frame : list of dict
    weights_per_frame : list of (N,) ndarrays
    frame_times : array-like
    save_path : str or None

    Returns
    -------
    matplotlib Figure
    """
    n_params = len(param_names)
    n_frames = len(params_per_frame)
    frame_times = np.asarray(frame_times)

    fig, axes = plt.subplots(n_params, 1, figsize=(8, 2.5 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for j, (name, ax) in enumerate(zip(param_names, axes)):
        means = []
        stds = []
        gt_vals = []

        for i in range(n_frames):
            w = weights_per_frame[i]
            w = w / w.sum()
            samples = params_per_frame[i][:, j]
            m = np.sum(w * samples)
            s = np.sqrt(np.sum(w * (samples - m) ** 2))
            means.append(m)
            stds.append(s)
            gt_vals.append(gt_per_frame[i][GT_KEYS[j]])

        means = np.array(means)
        stds = np.array(stds)

        ax.fill_between(frame_times, means - stds, means + stds,
                         alpha=0.25, color='steelblue', label='Posterior ±1σ')
        ax.plot(frame_times, means, 'o-', color='steelblue',
                ms=5, lw=1.5, label='Posterior mean')
        ax.plot(frame_times, gt_vals, 's--', color='red',
                ms=5, lw=1.5, label='Ground truth')

        ax.set_ylabel(name, fontsize=11)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc='best', fontsize=8)

    axes[-1].set_xlabel('Time (hr)', fontsize=11)
    fig.suptitle('Parameter Evolution Over Time', fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_frame_images(images, frame_times, pixel_size_uas=None,
                       gt_images=None, save_path=None):
    """
    Montage of posterior mean images across time frames.

    Parameters
    ----------
    images : list or (n_frames, npix, npix) ndarray
        Weighted mean image for each frame.
    frame_times : array-like
        Time in hours for each frame.
    pixel_size_uas : float or None
        Pixel size for axis labels.
    gt_images : list or None
        If given, show ground truth images in a second row.
    save_path : str or None

    Returns
    -------
    matplotlib Figure
    """
    images = np.asarray(images)
    n_frames = len(images)
    n_show = min(n_frames, 10)
    frame_indices = np.linspace(0, n_frames - 1, n_show, dtype=int)
    frame_times = np.asarray(frame_times)

    n_rows = 2 if gt_images is not None else 1
    fig, axes = plt.subplots(n_rows, n_show, figsize=(2.2 * n_show, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :] if n_show > 1 else np.array([[axes]])
    if n_show == 1:
        axes = axes[:, np.newaxis]

    npix = images.shape[1]
    if pixel_size_uas is not None:
        half = npix * pixel_size_uas / 2
        extent = [half, -half, -half, half]
    else:
        extent = None

    for col, idx in enumerate(frame_indices):
        ax = axes[0, col]
        ax.imshow(images[idx], origin='lower', cmap='afmhot', extent=extent)
        ax.set_title(f't={frame_times[idx]:.1f}h', fontsize=9)
        if col == 0:
            ax.set_ylabel('Posterior mean', fontsize=9)
        ax.tick_params(labelsize=7)

        if gt_images is not None:
            ax2 = axes[1, col]
            ax2.imshow(gt_images[idx], origin='lower', cmap='afmhot', extent=extent)
            if col == 0:
                ax2.set_ylabel('Ground truth', fontsize=9)
            ax2.tick_params(labelsize=7)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_loss_curves(loss_history, save_path=None):
    """
    Plot training loss curves for a single frame.

    Parameters
    ----------
    loss_history : dict
        Keys: 'total', 'cphase', 'logca'. Values: list of loss values per epoch.
    save_path : str or None

    Returns
    -------
    matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

    for ax, key, label in zip(axes,
                               ['total', 'cphase', 'logca'],
                               ['Total Loss', 'Closure Phase χ²', 'Log Closure Amp χ²']):
        vals = loss_history.get(key, [])
        if len(vals) > 0:
            ax.plot(vals, lw=0.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
            # Running average
            if len(vals) > 50:
                kernel = np.ones(50) / 50
                smooth = np.convolve(vals, kernel, mode='valid')
                ax.plot(np.arange(25, 25 + len(smooth)), smooth,
                        'r-', lw=1.5, label='50-epoch avg')
                ax.legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def compute_frame_metrics(params_per_frame, gt_per_frame, weights_per_frame,
                           param_names):
    """
    Compute per-frame parameter recovery metrics.

    Returns
    -------
    dict with keys 'means', 'stds', 'biases', 'gt_values' — each (n_frames, n_params).
    """
    n_frames = len(params_per_frame)
    n_params = len(param_names)

    means = np.zeros((n_frames, n_params))
    stds = np.zeros((n_frames, n_params))
    gt_values = np.zeros((n_frames, n_params))

    for i in range(n_frames):
        w = weights_per_frame[i]
        w = w / w.sum()
        for j in range(n_params):
            samples = params_per_frame[i][:, j]
            m = np.sum(w * samples)
            s = np.sqrt(np.sum(w * (samples - m) ** 2))
            means[i, j] = m
            stds[i, j] = s
            gt_values[i, j] = gt_per_frame[i][GT_KEYS[j]]

    biases = means - gt_values

    return {
        'means': means,
        'stds': stds,
        'biases': biases,
        'gt_values': gt_values,
        'param_names': param_names,
    }


def print_frame_metrics(metrics):
    """Print a summary table of per-frame metrics."""
    means = metrics['means']
    stds = metrics['stds']
    biases = metrics['biases']
    gt_values = metrics['gt_values']
    param_names = metrics['param_names']
    n_frames, n_params = means.shape

    print(f"{'Frame':>5s}", end='')
    for name in param_names:
        print(f"  {name:>20s}", end='')
    print()
    print("-" * (5 + 22 * n_params))

    for i in range(n_frames):
        print(f"{i:5d}", end='')
        for j in range(n_params):
            print(f"  {means[i,j]:7.2f}±{stds[i,j]:<5.2f}"
                  f"({biases[i,j]:+.1f})", end='')
        print()

    print("-" * (5 + 22 * n_params))
    print(f"{'Avg':>5s}", end='')
    for j in range(n_params):
        avg_bias = np.mean(np.abs(biases[:, j]))
        avg_std = np.mean(stds[:, j])
        print(f"  |bias|={avg_bias:<5.2f} σ={avg_std:<5.2f}   ", end='')
    print()
