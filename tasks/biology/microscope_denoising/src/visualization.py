"""
Visualization utilities and image quality metrics for microscope denoising.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_psnr(pred, gt, data_range=None):
    """
    Compute Peak Signal-to-Noise Ratio (dB).

    Parameters
    ----------
    pred, gt : np.ndarray
        Predicted and ground truth images (same shape, any numeric dtype).
    data_range : float or None
        Value range for PSNR. If None, uses gt.max() - gt.min().

    Returns
    -------
    float
    """
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    if data_range is None:
        data_range = gt.max() - gt.min()
    return peak_signal_noise_ratio(gt, pred, data_range=data_range)


def compute_ssim(pred, gt, data_range=None):
    """Compute Structural Similarity Index (SSIM)."""
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    if data_range is None:
        data_range = gt.max() - gt.min()
    return structural_similarity(gt, pred, data_range=data_range)


def compute_nrmse(pred, gt):
    """Compute Normalised Root Mean Square Error (lower = better)."""
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    return np.sqrt(np.mean((pred - gt) ** 2)) / (gt.max() - gt.min() + 1e-10)


def compute_snr_improvement(noisy, denoised, gt):
    """
    Compute SNR improvement: PSNR(denoised, gt) - PSNR(noisy, gt) in dB.
    """
    data_range = gt.max() - gt.min()
    psnr_noisy = compute_psnr(noisy, gt, data_range)
    psnr_denoised = compute_psnr(denoised, gt, data_range)
    return psnr_noisy, psnr_denoised, psnr_denoised - psnr_noisy


def compute_all_metrics(pred, gt):
    """
    Compute PSNR, SSIM, NRMSE for a prediction/ground truth pair.

    Returns
    -------
    dict with keys: psnr, ssim, nrmse
    """
    data_range = float(gt.max() - gt.min())
    return {
        'psnr': compute_psnr(pred, gt, data_range),
        'ssim': compute_ssim(pred, gt, data_range),
        'nrmse': compute_nrmse(pred, gt),
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(noisy, denoised, gt=None, titles=None,
                    pmin=1, pmax=99.5, figsize=None, save_path=None):
    """
    Side-by-side comparison: noisy input, denoised (Stage 1), and optionally
    a third image such as the Stage 2 deconvolved output.

    Parameters
    ----------
    noisy, denoised : np.ndarray, shape (H, W)
    gt : np.ndarray or None
        Third panel (deconvolved output or high-SNR reference).
    titles : list of str or None
    pmin, pmax : float
        Percentile clipping for display.
    save_path : str or None
    """
    images = [noisy, denoised] + ([gt] if gt is not None else [])
    n = len(images)
    if titles is None:
        titles = ['Noisy input', 'Stage 1: denoised']
        if gt is not None:
            titles.append('Stage 2: deconvolved')

    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        lo = np.percentile(img, pmin)
        hi = np.percentile(img, pmax)
        ax.imshow(img, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_zoom(noisy, denoised, gt=None, roi=None,
              pmin=1, pmax=99.5, figsize=None, save_path=None):
    """
    Show full image (top row) and zoomed-in ROI (bottom row).

    Parameters
    ----------
    roi : tuple (row_start, row_end, col_start, col_end) or None.
          If None, defaults to the centre quarter of the image.
    """
    H, W = noisy.shape
    if roi is None:
        roi = (H // 4, 3 * H // 4, W // 4, 3 * W // 4)
    r0, r1, c0, c1 = roi
    images_full = [noisy, denoised] + ([gt] if gt is not None else [])
    images_zoom = [img[r0:r1, c0:c1] for img in images_full]
    titles = ['Noisy', 'Denoised (Stage 1)'] + (['Deconvolved (Stage 2)'] if gt is not None else [])

    n = len(images_full)
    if figsize is None:
        figsize = (4 * n, 8)
    fig, axes = plt.subplots(2, n, figsize=figsize)

    for col, (img_f, img_z, title) in enumerate(zip(images_full, images_zoom, titles)):
        for row, img in enumerate([img_f, img_z]):
            ax = axes[row, col] if n > 1 else axes[row]
            lo = np.percentile(img_f, pmin)
            hi = np.percentile(img_f, pmax)
            ax.imshow(img, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
            label = title if row == 0 else f'{title} (zoom)'
            ax.set_title(label, fontsize=10)
            ax.axis('off')
        # Draw ROI rectangle on full image
        from matplotlib.patches import Rectangle
        ax_f = axes[0, col] if n > 1 else axes[0]
        rect = Rectangle((c0, r0), c1 - c0, r1 - r0,
                          linewidth=1.5, edgecolor='yellow', facecolor='none')
        ax_f.add_patch(rect)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def compute_psf_residual(deconvolved, noisy, psf):
    """
    PSF consistency metric: ||deconvolved * PSF - noisy||² / ||noisy||²

    Measures how well the deconvolved image, when re-blurred with the PSF,
    reconstructs the observed noisy image.  Lower = better consistency.

    Parameters
    ----------
    deconvolved : np.ndarray, shape (H, W)
    noisy : np.ndarray, shape (H, W)
    psf : np.ndarray, shape (kH, kW)

    Returns
    -------
    float
    """
    from scipy.ndimage import convolve
    blurred = convolve(deconvolved.astype(np.float64), psf.astype(np.float64),
                       mode='reflect')
    noisy_d = noisy.astype(np.float64)
    return float(np.mean((blurred - noisy_d) ** 2) / (np.mean(noisy_d ** 2) + 1e-10))


def plot_training_curve(loss_history, log_scale=True, save_path=None):
    """
    Plot training loss versus iteration.

    Parameters
    ----------
    loss_history : list of float  OR  list of (total, den, dec) tuples
        If tuples, all three components are plotted on the same axes.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    steps = np.arange(len(loss_history)) * 100

    if np.ndim(loss_history[0]) == 0:
        # Plain scalar list (Stage 1 only)
        ax.plot(steps, loss_history, linewidth=1.2, label='Total')
    else:
        arr = np.array(loss_history)
        ax.plot(steps, arr[:, 0], linewidth=1.4, label='Total')
        ax.plot(steps, arr[:, 1], linewidth=1.0, linestyle='--', label='Denoising (Stage 1)')
        ax.plot(steps, arr[:, 2], linewidth=1.0, linestyle=':', label='Deconvolution (Stage 2)')
        ax.legend(fontsize=9)

    for decay_step in [10000, 20000]:
        ax.axvline(decay_step, color='tomato', linestyle='--', alpha=0.5, linewidth=1)

    if log_scale:
        ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('ZS-DeconvNet joint training curve')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_intensity_profile(noisy, denoised, gt=None, row=None,
                           gt_label='Deconvolved (Stage 2)', save_path=None):
    """Plot a horizontal intensity profile through the image centre."""
    H = noisy.shape[0]
    if row is None:
        row = H // 2
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(noisy[row], color='gray', alpha=0.7, linewidth=1, label='Noisy')
    ax.plot(denoised[row], color='steelblue', linewidth=1.5, label='Denoised')
    if gt is not None:
        ax.plot(gt[row], color='tomato', linewidth=1.5, linestyle='--', label=gt_label)
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Intensity (ADU)')
    ax.set_title(f'Horizontal intensity profile (row {row})')
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
