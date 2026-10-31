"""
Visualization for Lucky Imaging
================================

Plotting utilities for comparing single frames, mean frames, and stacked
results. Quality metrics computation.

Functions
---------
plot_frame_comparison      Side-by-side single/mean/stacked comparison
plot_quality_histogram     Histogram of per-frame quality scores
plot_ap_grid               Overlay alignment point grid on image
plot_zoom_comparison       Zoomed crop comparison
compute_metrics            Compute quality metrics
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def _to_display(img):
    """Convert image to float [0,1] for matplotlib display."""
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype in (np.float32, np.float64):
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            return ((img - vmin) / (vmax - vmin)).astype(np.float32)
        return np.zeros_like(img, dtype=np.float32)
    return img.astype(np.float32)


def plot_frame_comparison(single_frame, mean_frame, stacked_frame,
                           titles=None, figsize=(18, 6)):
    """Plot side-by-side comparison of single, mean, and stacked images.

    Parameters
    ----------
    single_frame : ndarray -- best single frame
    mean_frame : ndarray -- simple average of all frames
    stacked_frame : ndarray -- lucky-imaging stacked result
    titles : list of str or None
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    if titles is None:
        titles = ["Best Single Frame", "Simple Mean", "Lucky Imaging Stack"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, img, title in zip(axes, [single_frame, mean_frame, stacked_frame],
                                titles):
        disp = _to_display(img)
        if disp.ndim == 2:
            ax.imshow(disp, cmap='gray')
        else:
            ax.imshow(disp)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    return fig


def plot_quality_histogram(quality_scores, selected_threshold=None,
                            figsize=(8, 4)):
    """Plot histogram of per-frame quality scores.

    Parameters
    ----------
    quality_scores : ndarray (N,) -- normalised quality scores
    selected_threshold : float or None -- draw threshold line
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(quality_scores, bins=20, color='steelblue', edgecolor='white',
            alpha=0.8)
    if selected_threshold is not None:
        ax.axvline(selected_threshold, color='red', linestyle='--',
                    label=f'Selection threshold ({selected_threshold:.2f})')
        ax.legend()
    ax.set_xlabel('Quality Score (normalised)')
    ax.set_ylabel('Number of Frames')
    ax.set_title('Frame Quality Distribution')
    fig.tight_layout()
    return fig


def plot_ap_grid(image, alignment_points, figsize=(12, 9)):
    """Overlay alignment point grid on an image.

    Parameters
    ----------
    image : ndarray
    alignment_points : list of dict with 'y', 'x', 'patch_*' keys
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    disp = _to_display(image)
    if disp.ndim == 2:
        ax.imshow(disp, cmap='gray')
    else:
        ax.imshow(disp)

    for ap in alignment_points:
        y_lo = ap['patch_y_low']
        y_hi = ap['patch_y_high']
        x_lo = ap['patch_x_low']
        x_hi = ap['patch_x_high']
        rect = plt.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                               linewidth=0.5, edgecolor='cyan',
                               facecolor='none', alpha=0.6)
        ax.add_patch(rect)
        ax.plot(ap['x'], ap['y'], 'r.', markersize=2)

    ax.set_title(f'Alignment Point Grid ({len(alignment_points)} APs)')
    ax.axis('off')
    fig.tight_layout()
    return fig


def plot_zoom_comparison(single_frame, stacked_frame, region,
                          titles=None, figsize=(14, 6)):
    """Show zoomed-in crop comparison.

    Parameters
    ----------
    single_frame : ndarray
    stacked_frame : ndarray
    region : tuple -- (y_low, y_high, x_low, x_high)
    titles : list of str or None
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    if titles is None:
        titles = ["Best Single Frame (Zoom)", "Lucky Imaging Stack (Zoom)"]

    y_lo, y_hi, x_lo, x_hi = region

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, img, title in zip(axes, [single_frame, stacked_frame], titles):
        crop = img[y_lo:y_hi, x_lo:x_hi]
        disp = _to_display(crop)
        if disp.ndim == 2:
            ax.imshow(disp, cmap='gray', interpolation='nearest')
        else:
            ax.imshow(disp, interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    return fig


def compute_metrics(stacked, reference_mean, best_frame, n_alignment_points=0,
                     n_frames_used=0):
    """Compute quality metrics comparing stacked result to baselines.

    Parameters
    ----------
    stacked : ndarray -- stacked image (uint16 or float)
    reference_mean : ndarray -- simple mean of all frames
    best_frame : ndarray -- single best frame
    n_alignment_points : int
    n_frames_used : int

    Returns
    -------
    metrics : dict
    """
    def _to_gray_f64(img):
        if img.dtype == np.uint16:
            img_f = img.astype(np.float64) / 65535.0
        elif img.dtype == np.uint8:
            img_f = img.astype(np.float64) / 255.0
        else:
            img_f = img.astype(np.float64)
        if img_f.ndim == 3:
            img_f = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + \
                    0.114 * img_f[:, :, 2]
        return img_f

    stacked_g = _to_gray_f64(stacked)
    mean_g = _to_gray_f64(reference_mean)
    best_g = _to_gray_f64(best_frame)

    # Resize to match if needed
    target_shape = stacked_g.shape
    if mean_g.shape != target_shape:
        mean_g = cv2.resize(mean_g, (target_shape[1], target_shape[0]))
    if best_g.shape != target_shape:
        best_g = cv2.resize(best_g, (target_shape[1], target_shape[0]))

    # Sharpness via Laplacian variance
    def laplacian_var(img):
        lap = cv2.Laplacian((img * 255).astype(np.uint8), cv2.CV_64F)
        return float(lap.var())

    sharp_stacked = laplacian_var(stacked_g)
    sharp_best = laplacian_var(best_g)
    sharp_mean = laplacian_var(mean_g)

    # SNR improvement estimate (based on noise variance reduction)
    noise_best = np.std(best_g - cv2.GaussianBlur(best_g, (5, 5), 0))
    noise_stacked = np.std(stacked_g - cv2.GaussianBlur(stacked_g, (5, 5), 0))
    snr_improvement = float(noise_best / max(noise_stacked, 1e-10))

    return {
        'sharpness_stacked': round(sharp_stacked, 4),
        'sharpness_best_frame': round(sharp_best, 4),
        'sharpness_mean': round(sharp_mean, 4),
        'sharpness_ratio_vs_best': round(sharp_stacked / max(sharp_best, 1e-10), 4),
        'sharpness_ratio_vs_mean': round(sharp_stacked / max(sharp_mean, 1e-10), 4),
        'snr_improvement': round(snr_improvement, 4),
        'n_alignment_points': n_alignment_points,
        'n_frames_used': n_frames_used,
    }
