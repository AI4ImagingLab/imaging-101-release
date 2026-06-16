"""
Visualization and metrics for sparse-view CT reconstruction.
"""

import numpy as np


def compute_nrmse(estimate, reference):
    """Compute NRMSE (normalized by dynamic range of reference).

    Parameters
    ----------
    estimate : np.ndarray
    reference : np.ndarray

    Returns
    -------
    nrmse : float
    """
    mse = np.mean((estimate - reference) ** 2)
    rmse = np.sqrt(mse)
    drange = np.max(reference) - np.min(reference)
    if drange == 0:
        return 0.0
    return float(rmse / drange)


def compute_ncc(estimate, reference):
    """Compute NCC (cosine similarity, no mean subtraction).

    Parameters
    ----------
    estimate : np.ndarray
    reference : np.ndarray

    Returns
    -------
    ncc : float
    """
    e = estimate.ravel().astype(np.float64)
    r = reference.ravel().astype(np.float64)
    norm_e = np.linalg.norm(e)
    norm_r = np.linalg.norm(r)
    if norm_e == 0 or norm_r == 0:
        return 0.0
    return float(np.dot(e, r) / (norm_e * norm_r))


def compute_ssim(estimate, reference):
    """Compute SSIM between two images.

    Parameters
    ----------
    estimate : np.ndarray, shape (H, W)
    reference : np.ndarray, shape (H, W)

    Returns
    -------
    ssim_val : float
    """
    from skimage.metrics import structural_similarity
    data_range = np.max(reference) - np.min(reference)
    return float(structural_similarity(estimate, reference, data_range=data_range))


def centre_crop(image, crop_fraction=0.8):
    """Extract centre crop of an image.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
    crop_fraction : float
        Fraction of image to keep (0..1).

    Returns
    -------
    cropped : np.ndarray
    """
    H, W = image.shape
    ch, cw = int(H * crop_fraction), int(W * crop_fraction)
    r0 = (H - ch) // 2
    c0 = (W - cw) // 2
    return image[r0:r0 + ch, c0:c0 + cw]


def plot_reconstruction_comparison(phantom, fbp_recon, tv_recon, save_path=None):
    """Plot ground truth, FBP, and TV reconstruction side by side.

    Parameters
    ----------
    phantom : np.ndarray, shape (H, W)
    fbp_recon : np.ndarray, shape (H, W)
    tv_recon : np.ndarray, shape (H, W)
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    vmin, vmax = phantom.min(), phantom.max()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(phantom, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(fbp_recon, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("FBP (sparse)")
    axes[1].axis("off")

    axes[2].imshow(tv_recon, cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].set_title("TV Reconstruction")
    axes[2].axis("off")

    # Error map for TV
    error = np.abs(tv_recon - phantom)
    axes[3].imshow(error, cmap="hot")
    axes[3].set_title("TV Error Map")
    axes[3].axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_sinograms(sinogram_full, sinogram_sparse, angles_full, angles_sparse,
                   save_path=None):
    """Plot full and sparse sinograms.

    Parameters
    ----------
    sinogram_full : np.ndarray, shape (n_det, n_angles_full)
    sinogram_sparse : np.ndarray, shape (n_det, n_angles_sparse)
    angles_full : np.ndarray
    angles_sparse : np.ndarray
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(sinogram_full, aspect="auto", cmap="gray",
                   extent=[angles_full[0], angles_full[-1], sinogram_full.shape[0], 0])
    axes[0].set_title(f"Full Sinogram ({len(angles_full)} angles)")
    axes[0].set_xlabel("Angle (deg)")
    axes[0].set_ylabel("Detector index")

    axes[1].imshow(sinogram_sparse, aspect="auto", cmap="gray",
                   extent=[angles_sparse[0], angles_sparse[-1], sinogram_sparse.shape[0], 0])
    axes[1].set_title(f"Sparse Sinogram ({len(angles_sparse)} angles)")
    axes[1].set_xlabel("Angle (deg)")
    axes[1].set_ylabel("Detector index")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_loss_history(loss_history, save_path=None):
    """Plot convergence of the TV solver.

    Parameters
    ----------
    loss_history : list of float
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(loss_history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Data Fidelity Loss")
    ax.set_title("TV Reconstruction Convergence")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
