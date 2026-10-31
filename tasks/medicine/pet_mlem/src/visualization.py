"""
Visualization and metrics for PET MLEM reconstruction.
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


def plot_pet_reconstruction(gt, recon_mlem, recon_osem, save_path=None):
    """Plot ground truth vs MLEM vs OSEM reconstructions."""
    import matplotlib.pyplot as plt

    vmin, vmax = 0, gt.max()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(gt, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth Activity')
    axes[0].axis('off')

    axes[1].imshow(recon_mlem, cmap='hot', vmin=vmin, vmax=vmax)
    axes[1].set_title('MLEM Reconstruction')
    axes[1].axis('off')

    axes[2].imshow(recon_osem, cmap='hot', vmin=vmin, vmax=vmax)
    axes[2].set_title('OSEM Reconstruction')
    axes[2].axis('off')

    error = np.abs(recon_osem - gt)
    im = axes[3].imshow(error, cmap='viridis', vmin=0, vmax=vmax * 0.3)
    axes[3].set_title('OSEM Absolute Error')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_convergence(ll_mlem, ll_osem, save_path=None):
    """Plot log-likelihood convergence for MLEM and OSEM."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ll_mlem, 'b-', label='MLEM')
    ax.plot(np.arange(len(ll_osem)) * (len(ll_mlem) / len(ll_osem)),
            ll_osem, 'r-o', label='OSEM')
    ax.set_xlabel('Iteration (MLEM equivalent)')
    ax.set_ylabel('Poisson Log-Likelihood')
    ax.set_title('Convergence: MLEM vs OSEM')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_sinogram(sinogram, title="PET Sinogram", save_path=None):
    """Plot a PET sinogram."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(sinogram, aspect='auto', cmap='hot')
    ax.set_xlabel('Projection angle index')
    ax.set_ylabel('Radial bin')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig
