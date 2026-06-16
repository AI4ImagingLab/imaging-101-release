"""
Visualization and Metrics for Closure-Only VLBI Imaging
========================================================

Plotting utilities and image quality metrics.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(estimate: np.ndarray, reference: np.ndarray) -> dict:
    """
    Compute image quality metrics after flux normalization.

    The estimate is flux-normalized to match the reference total flux
    before computing metrics.

    Parameters
    ----------
    estimate  : (N, N) reconstructed image
    reference : (N, N) ground truth image

    Returns
    -------
    dict with 'nrmse', 'ncc', 'dynamic_range'
    """
    est = estimate.copy()
    ref = reference.copy()

    # Flux normalization
    est *= ref.sum() / (est.sum() + 1e-30)

    # NRMSE (RMS error normalised by dynamic range of reference)
    nrmse = float(np.sqrt(np.mean((est - ref)**2))
                  / (ref.max() - ref.min() + 1e-30))

    # Normalized Cross-Correlation
    ncc = float(np.sum(est * ref)
               / (np.sqrt(np.sum(est**2)) * np.sqrt(np.sum(ref**2)) + 1e-30))

    # Dynamic range
    peak = est.max()
    residual_rms = np.sqrt(np.mean((est - ref)**2)) + 1e-30
    dynamic_range = float(peak / residual_rms)

    return {
        'nrmse': round(nrmse, 4),
        'ncc': round(ncc, 4),
        'dynamic_range': round(dynamic_range, 4),
    }


def print_metrics_table(metrics_dict: dict):
    """Print a formatted table of metrics."""
    print(f"{'Method':<30s} {'NRMSE':>8s} {'NCC':>8s} {'DynRange':>10s}")
    print('-' * 58)
    for name, m in metrics_dict.items():
        print(f"{name:<30s} {m['nrmse']:>8.4f} {m['ncc']:>8.4f} {m['dynamic_range']:>10.2f}")


def plot_uv_coverage(uv_coords: np.ndarray, title: str = 'UV Coverage',
                     ax=None):
    """Plot (u,v) coverage in Gλ."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5))

    u_gl = uv_coords[:, 0] / 1e9
    v_gl = uv_coords[:, 1] / 1e9
    ax.scatter(u_gl, v_gl, s=6, c='steelblue', alpha=0.7, label='measured')
    ax.scatter(-u_gl, -v_gl, s=6, c='salmon', alpha=0.7, label='conjugate')
    ax.set_xlabel('u (Gλ)')
    ax.set_ylabel('v (Gλ)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', lw=0.4, ls='--')
    ax.axvline(0, color='k', lw=0.4, ls='--')
    ax.legend(fontsize=8, markerscale=2)
    return ax


def plot_image(image: np.ndarray, pixel_size_uas: float = 2.0,
               title: str = '', cmap: str = 'afmhot', ax=None, vmin=0, vmax=None):
    """Plot a single image."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))

    N = image.shape[0]
    fov = N * pixel_size_uas
    hw = fov / 2

    if vmax is None:
        vmax = image.max()

    im = ax.imshow(image, cmap=cmap, origin='lower',
                   extent=[-hw, hw, -hw, hw], vmin=vmin, vmax=vmax)
    ax.set_xlabel('Relative RA (μas)')
    ax.set_ylabel('Relative Dec (μas)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_comparison(images: dict, ground_truth: np.ndarray,
                    metrics_dict: dict = None,
                    pixel_size_uas: float = 2.0,
                    suptitle: str = 'Reconstruction Comparison'):
    """
    Side-by-side comparison of reconstructions.

    Parameters
    ----------
    images : dict mapping method name → (N,N) image
    ground_truth : (N,N) reference image
    metrics_dict : dict mapping method name → metrics dict
    """
    N = ground_truth.shape[0]
    fov = N * pixel_size_uas
    hw = fov / 2

    panels = [('Ground Truth', ground_truth)] + list(images.items())
    n = len(panels)

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap='afmhot', origin='lower',
                  extent=[-hw, hw, -hw, hw], vmin=0, vmax=img.max())
        if metrics_dict and title in metrics_dict:
            m = metrics_dict[title]
            title = f'{title}\nNRMSE={m["nrmse"]:.3f} NCC={m["ncc"]:.3f}'
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('μas')
        ax.set_ylabel('μas')

    fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_gain_robustness(cal_metrics: dict, cor_metrics: dict):
    """Print calibrated vs corrupted metrics table."""
    print(f"{'Method':<25s} {'Calibrated':>18s} {'Corrupted':>18s}")
    print(f"{'':25s} {'NRMSE':>8s} {'NCC':>8s}   {'NRMSE':>8s} {'NCC':>8s}")
    print('-' * 63)
    for name in cal_metrics:
        mc = cal_metrics[name]
        mg = cor_metrics.get(name, {'nrmse': float('nan'), 'ncc': float('nan')})
        print(f"{name:<25s} {mc['nrmse']:>8.4f} {mc['ncc']:>8.4f}   "
              f"{mg['nrmse']:>8.4f} {mg['ncc']:>8.4f}")
