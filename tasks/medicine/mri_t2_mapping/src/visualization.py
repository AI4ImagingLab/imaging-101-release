"""
Visualization and metrics for MRI T2 mapping.
"""

import numpy as np


def compute_ncc(estimate, reference, mask=None):
    """Compute Normalized Cross-Correlation (cosine similarity).

    NCC = (x_hat . x_ref) / (||x_hat|| * ||x_ref||)

    Parameters
    ----------
    estimate : np.ndarray
        Estimated map.
    reference : np.ndarray
        Reference (ground truth) map.
    mask : np.ndarray or None
        Boolean mask. If provided, only masked pixels are used.

    Returns
    -------
    ncc : float
        NCC value in [0, 1] for non-negative signals.
    """
    if mask is not None:
        estimate = estimate[mask]
        reference = reference[mask]

    estimate = estimate.ravel().astype(np.float64)
    reference = reference.ravel().astype(np.float64)

    norm_est = np.linalg.norm(estimate)
    norm_ref = np.linalg.norm(reference)

    if norm_est < 1e-12 or norm_ref < 1e-12:
        return 0.0

    return float(np.dot(estimate, reference) / (norm_est * norm_ref))


def compute_nrmse(estimate, reference, mask=None):
    """Compute Normalized Root Mean Square Error.

    NRMSE = sqrt(mean((x_hat - x_ref)^2)) / (max(x_ref) - min(x_ref))

    Parameters
    ----------
    estimate : np.ndarray
        Estimated map.
    reference : np.ndarray
        Reference (ground truth) map.
    mask : np.ndarray or None
        Boolean mask.

    Returns
    -------
    nrmse : float
        NRMSE value (lower is better).
    """
    if mask is not None:
        estimate = estimate[mask]
        reference = reference[mask]

    estimate = estimate.ravel().astype(np.float64)
    reference = reference.ravel().astype(np.float64)

    dynamic_range = reference.max() - reference.min()
    if dynamic_range < 1e-12:
        return float('inf')

    rmse = np.sqrt(np.mean((estimate - reference) ** 2))
    return float(rmse / dynamic_range)


def plot_t2_maps(T2_gt, T2_est, tissue_mask, title_est="Estimated T2",
                 vmin=0, vmax=200, save_path=None):
    """Plot ground truth and estimated T2 maps side by side.

    Parameters
    ----------
    T2_gt : np.ndarray
        Ground truth T2 map, shape (Ny, Nx).
    T2_est : np.ndarray
        Estimated T2 map, shape (Ny, Nx).
    tissue_mask : np.ndarray
        Boolean tissue mask, shape (Ny, Nx).
    title_est : str
        Title for the estimated map.
    vmin, vmax : float
        Colorbar range in ms.
    save_path : str or None
        If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    T2_gt_disp = np.where(tissue_mask, T2_gt, np.nan)
    T2_est_disp = np.where(tissue_mask, T2_est, np.nan)
    error = np.where(tissue_mask, np.abs(T2_est - T2_gt), np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(T2_gt_disp, vmin=vmin, vmax=vmax, cmap='hot')
    axes[0].set_title('Ground Truth T2 (ms)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(T2_est_disp, vmin=vmin, vmax=vmax, cmap='hot')
    axes[1].set_title(f'{title_est} (ms)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(error, vmin=0, vmax=vmax * 0.3, cmap='viridis')
    axes[2].set_title('Absolute Error (ms)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_signal_decay(signal, TE, T2_gt, T2_est, pixel_coords, save_path=None):
    """Plot signal decay curves at selected pixels.

    Parameters
    ----------
    signal : np.ndarray
        Multi-echo signal, shape (Ny, Nx, N_echoes).
    TE : np.ndarray
        Echo times in ms.
    T2_gt : np.ndarray
        Ground truth T2, shape (Ny, Nx).
    T2_est : np.ndarray
        Estimated T2, shape (Ny, Nx).
    pixel_coords : list of (y, x) tuples
        Pixel locations to plot.
    save_path : str or None
        If provided, save figure.
    """
    import matplotlib.pyplot as plt

    n_pixels = len(pixel_coords)
    fig, axes = plt.subplots(1, n_pixels, figsize=(5 * n_pixels, 4))
    if n_pixels == 1:
        axes = [axes]

    TE_fine = np.linspace(TE[0], TE[-1], 200)

    for i, (y, x) in enumerate(pixel_coords):
        s = signal[y, x, :]
        t2_true = T2_gt[y, x]
        t2_fit = T2_est[y, x]
        m0_fit = s[0] * np.exp(TE[0] / t2_fit) if t2_fit > 0 else s[0]

        axes[i].plot(TE, s, 'ko', markersize=6, label='Data')
        axes[i].plot(TE_fine, m0_fit * np.exp(-TE_fine / t2_true),
                     'b--', label=f'True T2={t2_true:.0f} ms')
        if t2_fit > 0:
            axes[i].plot(TE_fine, m0_fit * np.exp(-TE_fine / t2_fit),
                         'r-', label=f'Fit T2={t2_fit:.1f} ms')
        axes[i].set_xlabel('TE (ms)')
        axes[i].set_ylabel('Signal')
        axes[i].set_title(f'Pixel ({y}, {x})')
        axes[i].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig
