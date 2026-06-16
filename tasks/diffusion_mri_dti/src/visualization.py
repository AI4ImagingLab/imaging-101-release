"""
Visualization and metrics for diffusion MRI DTI.
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
        Boolean mask.

    Returns
    -------
    ncc : float
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


def plot_dti_maps(fa_gt, fa_est, md_gt, md_est, tissue_mask,
                  title_est="Estimated", save_path=None):
    """Plot ground truth vs estimated FA and MD maps.

    Parameters
    ----------
    fa_gt, fa_est : np.ndarray
        FA maps, shape (Ny, Nx).
    md_gt, md_est : np.ndarray
        MD maps in mm^2/s, shape (Ny, Nx).
    tissue_mask : np.ndarray
        Boolean mask, shape (Ny, Nx).
    title_est : str
        Label for estimated maps.
    save_path : str or None
        If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    fa_gt_d = np.where(tissue_mask, fa_gt, np.nan)
    fa_est_d = np.where(tissue_mask, fa_est, np.nan)
    md_gt_d = np.where(tissue_mask, md_gt * 1e3, np.nan)  # convert to 10^-3 mm^2/s
    md_est_d = np.where(tissue_mask, md_est * 1e3, np.nan)
    fa_err = np.where(tissue_mask, np.abs(fa_est - fa_gt), np.nan)
    md_err = np.where(tissue_mask, np.abs(md_est - md_gt) * 1e3, np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # FA row
    im = axes[0, 0].imshow(fa_gt_d, vmin=0, vmax=1, cmap='hot')
    axes[0, 0].set_title('Ground Truth FA')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = axes[0, 1].imshow(fa_est_d, vmin=0, vmax=1, cmap='hot')
    axes[0, 1].set_title(f'{title_est} FA')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[0, 2].imshow(fa_err, vmin=0, vmax=0.3, cmap='viridis')
    axes[0, 2].set_title('FA Absolute Error')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # MD row
    im = axes[1, 0].imshow(md_gt_d, vmin=0, vmax=3.5, cmap='hot')
    axes[1, 0].set_title(r'Ground Truth MD ($\times 10^{-3}$ mm$^2$/s)')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(md_est_d, vmin=0, vmax=3.5, cmap='hot')
    axes[1, 1].set_title(fr'{title_est} MD ($\times 10^{{-3}}$ mm$^2$/s)')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    im = axes[1, 2].imshow(md_err, vmin=0, vmax=0.5, cmap='viridis')
    axes[1, 2].set_title(r'MD Absolute Error ($\times 10^{-3}$ mm$^2$/s)')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_color_fa(fa_map, eigenvectors, tissue_mask, save_path=None):
    """Plot directionally-encoded color FA map.

    RGB channels encode the primary eigenvector direction:
    R = |v1_x| (left-right), G = |v1_y| (anterior-posterior), B = |v1_z| (superior-inferior)
    Brightness is modulated by FA.

    Parameters
    ----------
    fa_map : np.ndarray
        FA map, shape (Ny, Nx).
    eigenvectors : np.ndarray
        Eigenvectors, shape (Ny, Nx, 3, 3).
    tissue_mask : np.ndarray
        Boolean mask.
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    Ny, Nx = fa_map.shape
    color_fa = np.zeros((Ny, Nx, 3), dtype=np.float64)

    # Primary eigenvector (first column after sorting descending)
    v1 = eigenvectors[..., :, 0]  # (Ny, Nx, 3)
    v1_abs = np.abs(v1)

    # Normalize direction vector to unit length
    v1_norm = np.linalg.norm(v1_abs, axis=-1, keepdims=True)
    v1_norm = np.maximum(v1_norm, 1e-10)
    v1_abs = v1_abs / v1_norm

    # Color FA = |v1| * FA
    color_fa = v1_abs * fa_map[..., np.newaxis]
    color_fa = np.where(tissue_mask[..., np.newaxis], color_fa, 0.0)
    color_fa = np.clip(color_fa, 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(color_fa)
    ax.set_title('Directional Color FA Map\n(R=L-R, G=A-P, B=S-I)')
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig
