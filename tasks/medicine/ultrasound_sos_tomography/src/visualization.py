"""
Visualization and metrics for ultrasound speed-of-sound tomography.
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


def plot_sos_comparison(sos_gt, sos_fbp, sos_tv, save_path=None):
    """Plot ground truth, FBP, and TV-ADMM speed-of-sound maps side by side.

    Parameters
    ----------
    sos_gt : np.ndarray, shape (H, W)
        Ground truth SoS in m/s.
    sos_fbp : np.ndarray, shape (H, W)
        FBP reconstructed SoS.
    sos_tv : np.ndarray, shape (H, W)
        TV-ADMM reconstructed SoS.
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    vmin, vmax = sos_gt.min(), sos_gt.max()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    im0 = axes[0].imshow(sos_gt, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth SoS")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], shrink=0.7, label="m/s")

    im1 = axes[1].imshow(sos_fbp, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("FBP Reconstruction")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], shrink=0.7, label="m/s")

    im2 = axes[2].imshow(sos_tv, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].set_title("TV-ADMM Reconstruction")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], shrink=0.7, label="m/s")

    # Error map for TV
    error = np.abs(sos_tv - sos_gt)
    im3 = axes[3].imshow(error, cmap="hot")
    axes[3].set_title("TV-ADMM Error Map")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], shrink=0.7, label="m/s")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_sinograms(sinogram, sinogram_full, angles, angles_full, save_path=None):
    """Plot sparse and full travel-time sinograms.

    Parameters
    ----------
    sinogram : np.ndarray, shape (n_det, n_angles)
    sinogram_full : np.ndarray, shape (n_det, n_angles_full)
    angles : np.ndarray
    angles_full : np.ndarray
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(sinogram_full, aspect="auto", cmap="inferno",
                   extent=[angles_full[0], angles_full[-1],
                           sinogram_full.shape[0], 0])
    axes[0].set_title(f"Full Sinogram ({len(angles_full)} angles)")
    axes[0].set_xlabel("Angle (deg)")
    axes[0].set_ylabel("Detector index")

    axes[1].imshow(sinogram, aspect="auto", cmap="inferno",
                   extent=[angles[0], angles[-1],
                           sinogram.shape[0], 0])
    axes[1].set_title(f"Sparse Sinogram ({len(angles)} angles)")
    axes[1].set_xlabel("Angle (deg)")
    axes[1].set_ylabel("Detector index")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_loss_history(loss_history_sart, loss_history_tv, save_path=None):
    """Plot convergence of SART and TV-ADMM solvers.

    Parameters
    ----------
    loss_history_sart : list of float
    loss_history_tv : list of float
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(loss_history_sart, label="SART")
    ax.semilogy(loss_history_tv, label="TV-ADMM")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Data Fidelity Loss")
    ax.set_title("Reconstruction Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_sos_profiles(sos_gt, sos_fbp, sos_tv, row=None, save_path=None):
    """Plot horizontal line profiles through the SoS maps.

    Parameters
    ----------
    sos_gt, sos_fbp, sos_tv : np.ndarray, shape (H, W)
    row : int or None
        Row index for the profile. If None, uses the center row.
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    if row is None:
        row = sos_gt.shape[0] // 2

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sos_gt[row, :], "k-", linewidth=2, label="Ground Truth")
    ax.plot(sos_fbp[row, :], "b--", linewidth=1.5, label="FBP")
    ax.plot(sos_tv[row, :], "r-.", linewidth=1.5, label="TV-ADMM")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Speed of Sound (m/s)")
    ax.set_title(f"Horizontal Profile (row {row})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
