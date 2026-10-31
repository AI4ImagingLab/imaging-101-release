"""
DCE-MRI Visualization and Metrics
===================================

Provides:
- Per-frame and average metrics (NRMSE, NCC, PSNR)
- Frame comparison plots
- Time-activity curve plots
- Convergence plots
"""

import numpy as np


# ── Metrics ────────────────────────────────────────────────────────────────


def compute_nrmse(recon, reference):
    """
    Normalized root mean square error.

    NRMSE = sqrt(mean((recon - ref)^2)) / (max(ref) - min(ref))

    Parameters
    ----------
    recon : ndarray
        Reconstructed image(s).
    reference : ndarray
        Reference image(s), same shape.

    Returns
    -------
    float
        NRMSE value.
    """
    drange = reference.max() - reference.min()
    if drange < 1e-12:
        return 0.0
    rmse = np.sqrt(np.mean((recon - reference) ** 2))
    return float(rmse / drange)


def compute_ncc(recon, reference):
    """
    Normalized cross-correlation (cosine similarity).

    NCC = (recon . ref) / (||recon|| * ||ref||)

    Parameters
    ----------
    recon : ndarray
        Reconstructed image(s).
    reference : ndarray
        Reference image(s).

    Returns
    -------
    float
        NCC value in [-1, 1].
    """
    r = recon.ravel()
    g = reference.ravel()
    norm_r = np.linalg.norm(r)
    norm_g = np.linalg.norm(g)
    if norm_r < 1e-12 or norm_g < 1e-12:
        return 0.0
    return float(np.dot(r, g) / (norm_r * norm_g))


def compute_psnr(recon, reference):
    """
    Peak signal-to-noise ratio in dB.

    Parameters
    ----------
    recon, reference : ndarray

    Returns
    -------
    float
        PSNR in dB.
    """
    mse = np.mean((recon - reference) ** 2)
    if mse < 1e-30:
        return 100.0
    peak = reference.max()
    return float(10 * np.log10(peak ** 2 / mse))


def compute_frame_metrics(recon, reference):
    """
    Compute per-frame metrics for a dynamic reconstruction.

    Parameters
    ----------
    recon : ndarray, (T, N, N)
        Reconstructed time series.
    reference : ndarray, (T, N, N)
        Ground truth time series.

    Returns
    -------
    metrics : dict
        'per_frame': list of dicts with nrmse, ncc, psnr per frame
        'avg_nrmse', 'avg_ncc', 'avg_psnr': averages over frames
        'overall_nrmse', 'overall_ncc': computed on full 3D volume
    """
    T = recon.shape[0]
    per_frame = []
    for t in range(T):
        per_frame.append({
            'frame': t,
            'nrmse': compute_nrmse(recon[t], reference[t]),
            'ncc': compute_ncc(recon[t], reference[t]),
            'psnr': compute_psnr(recon[t], reference[t]),
        })

    avg_nrmse = np.mean([m['nrmse'] for m in per_frame])
    avg_ncc = np.mean([m['ncc'] for m in per_frame])
    avg_psnr = np.mean([m['psnr'] for m in per_frame])

    return {
        'per_frame': per_frame,
        'avg_nrmse': float(avg_nrmse),
        'avg_ncc': float(avg_ncc),
        'avg_psnr': float(avg_psnr),
        'overall_nrmse': compute_nrmse(recon, reference),
        'overall_ncc': compute_ncc(recon, reference),
    }


def print_metrics_table(metrics):
    """Print a summary table of dynamic reconstruction metrics."""
    print(f"  {'Frame':>5s}  {'NRMSE':>8s}  {'NCC':>8s}  {'PSNR':>8s}")
    print(f"  {'-----':>5s}  {'-----':>8s}  {'---':>8s}  {'----':>8s}")
    for m in metrics['per_frame']:
        print(f"  {m['frame']:5d}  {m['nrmse']:8.4f}  {m['ncc']:8.4f}  {m['psnr']:8.2f}")
    print(f"  {'Avg':>5s}  {metrics['avg_nrmse']:8.4f}  "
          f"{metrics['avg_ncc']:8.4f}  {metrics['avg_psnr']:8.2f}")


# ── Plotting ───────────────────────────────────────────────────────────────


def plot_frame_comparison(gt, zero_fill, tv_recon, frames=None, save_path=None):
    """
    Plot selected frames: ground truth, zero-fill, TV recon.

    Parameters
    ----------
    gt : ndarray, (T, N, N)
    zero_fill : ndarray, (T, N, N)
    tv_recon : ndarray, (T, N, N)
    frames : list of int or None
        Which frames to show. Default: evenly spaced 5 frames.
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    T = gt.shape[0]
    if frames is None:
        frames = np.linspace(0, T - 1, min(5, T), dtype=int)
    n_frames = len(frames)

    fig, axes = plt.subplots(3, n_frames, figsize=(3 * n_frames, 9))
    if n_frames == 1:
        axes = axes[:, None]

    vmin, vmax = gt.min(), gt.max()

    for i, t in enumerate(frames):
        axes[0, i].imshow(gt[t], cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f't={t}')
        axes[0, i].axis('off')

        axes[1, i].imshow(zero_fill[t], cmap='gray', vmin=vmin, vmax=vmax)
        axes[1, i].axis('off')

        axes[2, i].imshow(tv_recon[t], cmap='gray', vmin=vmin, vmax=vmax)
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel('Ground Truth', fontsize=12)
    axes[1, 0].set_ylabel('Zero-Fill', fontsize=12)
    axes[2, 0].set_ylabel('Temporal TV', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_time_activity_curves(gt, zero_fill, tv_recon, time_points,
                              roi_center, roi_radius=3, save_path=None):
    """
    Plot time-activity curves for a region of interest.

    Parameters
    ----------
    gt, zero_fill, tv_recon : ndarray, (T, N, N)
    time_points : ndarray, (T,)
    roi_center : tuple (row, col)
        Center of the ROI.
    roi_radius : int
        Half-size of the square ROI.
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    r, c = roi_center
    r0, r1 = max(0, r - roi_radius), r + roi_radius + 1
    c0, c1 = max(0, c - roi_radius), c + roi_radius + 1

    gt_curve = gt[:, r0:r1, c0:c1].mean(axis=(1, 2))
    zf_curve = zero_fill[:, r0:r1, c0:c1].mean(axis=(1, 2))
    tv_curve = tv_recon[:, r0:r1, c0:c1].mean(axis=(1, 2))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_points, gt_curve, 'k-o', label='Ground Truth', markersize=4)
    ax.plot(time_points, zf_curve, 'b--s', label='Zero-Fill', markersize=3, alpha=0.7)
    ax.plot(time_points, tv_curve, 'r-^', label='Temporal TV', markersize=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal Intensity')
    ax.set_title(f'Time-Activity Curve (ROI at [{r0}:{r1}, {c0}:{c1}])')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_convergence(loss_history, save_path=None):
    """
    Plot ADMM convergence (loss vs iteration).

    Parameters
    ----------
    loss_history : list of float
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(loss_history)
    ax.set_xlabel('ADMM Iteration')
    ax.set_ylabel('Loss (data fidelity + TV)')
    ax.set_title('ADMM Convergence')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_error_maps(gt, tv_recon, frames=None, save_path=None):
    """
    Plot absolute error maps for selected frames.

    Parameters
    ----------
    gt : ndarray, (T, N, N)
    tv_recon : ndarray, (T, N, N)
    frames : list of int or None
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    T = gt.shape[0]
    if frames is None:
        frames = np.linspace(0, T - 1, min(5, T), dtype=int)
    n_frames = len(frames)

    fig, axes = plt.subplots(1, n_frames, figsize=(3 * n_frames, 3))
    if n_frames == 1:
        axes = [axes]

    error = np.abs(gt - tv_recon)
    vmax_err = error.max()

    for i, t in enumerate(frames):
        im = axes[i].imshow(error[t], cmap='hot', vmin=0, vmax=vmax_err)
        axes[i].set_title(f't={t}')
        axes[i].axis('off')

    fig.colorbar(im, ax=axes, shrink=0.8, label='Absolute Error')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
