"""
Visualization and metrics for fan-beam CT.
"""

import numpy as np


def compute_ncc(estimate, reference, mask=None):
    """Compute Normalized Cross-Correlation (cosine similarity)."""
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
    """Compute Normalized Root Mean Square Error."""
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


def centre_crop_normalize(image, crop_fraction=0.8):
    """Centre-crop and normalize image to [0, 1].

    Parameters
    ----------
    image : np.ndarray, shape (N, N)
    crop_fraction : float

    Returns
    -------
    cropped : np.ndarray
    """
    N = image.shape[0]
    margin = int(N * (1 - crop_fraction) / 2)
    cropped = image[margin:N - margin, margin:N - margin].copy()
    vmin, vmax = cropped.min(), cropped.max()
    if vmax - vmin > 1e-12:
        cropped = (cropped - vmin) / (vmax - vmin)
    return cropped


def plot_reconstructions(phantom, recon_fbp, recon_tv, title_suffix="",
                         save_path=None):
    """Plot ground truth vs FBP vs TV reconstructions.

    Parameters
    ----------
    phantom : np.ndarray, shape (N, N)
    recon_fbp : np.ndarray, shape (N, N)
    recon_tv : np.ndarray, shape (N, N)
    title_suffix : str
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    vmin, vmax = phantom.min(), phantom.max()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(phantom, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    axes[1].imshow(recon_fbp, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'FBP {title_suffix}')
    axes[1].axis('off')

    axes[2].imshow(recon_tv, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title(f'TV-PDHG {title_suffix}')
    axes[2].axis('off')

    error = np.abs(recon_tv - phantom)
    im = axes[3].imshow(error, cmap='hot', vmin=0, vmax=vmax * 0.2)
    axes[3].set_title(f'TV Error {title_suffix}')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_sinogram(sinogram, title="Sinogram", save_path=None):
    """Plot a sinogram."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(sinogram, aspect='auto', cmap='gray')
    ax.set_xlabel('Detector element')
    ax.set_ylabel('Projection angle index')
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig
