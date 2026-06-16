"""Plotting helpers for Hessian-SIM reconstructions."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
from scipy.ndimage import zoom as ndizoom


def upsample_widefield(widefield, target_shape):
    """Upsample a widefield image (2x) and centre-crop/pad to ``target_shape``."""
    img = ndizoom(widefield, 2, order=3)
    ty, tx = target_shape
    sy, sx = img.shape
    if sy != ty or sx != tx:
        cy = (sy - ty) // 2
        cx = (sx - tx) // 2
        if cy >= 0 and cx >= 0:
            img = img[cy:cy + ty, cx:cx + tx]
        else:
            out = np.zeros(target_shape)
            cy2 = (ty - sy) // 2
            cx2 = (tx - sx) // 2
            out[cy2:cy2 + sy, cx2:cx2 + sx] = img
            img = out
    return img


def compute_widefield(raw_frames, nangles, nphases):
    """Incoherent average of all raw frames (no structured illumination)."""
    nframes_per_t = nangles * nphases
    nt = raw_frames.shape[0] // nframes_per_t
    return raw_frames[:nt * nframes_per_t].reshape(
        nt, nframes_per_t, raw_frames.shape[1], raw_frames.shape[2]
    ).mean(axis=(0, 1))


def plot_comparison(wf, wiener, hessian, tv, out_path):
    """Four-panel comparison: widefield | Wiener | Hessian | TV, plus FFT row."""
    images = [wf, wiener, hessian, tv]
    titles = ['Widefield', 'Wiener-SIM', 'Hessian-SIM', 'TV-SIM']
    vmax = np.percentile(wiener[wiener > 0], 99.5) if wiener.max() > 0 else 1

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[0, i].imshow(img, cmap='hot', vmin=0, vmax=vmax, interpolation='nearest')
        axes[0, i].set_title(title, fontsize=14, fontweight='bold')
        axes[0, i].axis('off')

    fts = [np.log1p(np.abs(fftshift(fft2(img)))) for img in images]
    ft_vmax = max(ft.max() for ft in fts)
    ft_vmin = min(ft.min() for ft in fts)
    for i, (ft, title) in enumerate(zip(fts, titles)):
        axes[1, i].imshow(ft, cmap='viridis', vmin=ft_vmin, vmax=ft_vmax, interpolation='nearest')
        axes[1, i].set_title(f'{title} – FFT', fontsize=12)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_line_profiles(wiener, hessian, tv, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    row = wiener.shape[0] // 2
    col = wiener.shape[1] // 2
    for img, title, c in zip(
        [wiener, hessian, tv],
        ['Wiener-SIM', 'Hessian-SIM', 'TV-SIM'],
        ['tab:blue', 'tab:orange', 'tab:green'],
    ):
        axes[0].plot(img[row, :], label=title, alpha=0.8, color=c)
        axes[1].plot(img[:, col], label=title, alpha=0.8, color=c)
    axes[0].set_title('Horizontal line profile (centre row)')
    axes[1].set_title('Vertical line profile (centre column)')
    for ax in axes:
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Intensity')
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def ncc(a, b):
    """Cosine-similarity NCC between two arrays (flattened)."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def nrmse(est, ref):
    """Dynamic-range normalised RMSE."""
    est = est.astype(np.float64)
    ref = ref.astype(np.float64)
    rng = ref.max() - ref.min()
    if rng == 0:
        return 0.0
    return float(np.sqrt(np.mean((est - ref) ** 2)) / rng)
