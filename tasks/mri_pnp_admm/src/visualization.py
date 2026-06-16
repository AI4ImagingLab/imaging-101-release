"""
Visualization and Metrics for PnP-ADMM CS-MRI Reconstruction
================================================================

Provides signal-power PSNR (matching the original paper), standard
NCC/NRMSE metrics (matching the benchmark harness), and plotting
utilities.
"""

import numpy as np


def compute_psnr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute signal-power PSNR (as defined in the original paper).

    PSNR = 10 * log10(||x_ref||^2 / ||x_est - x_ref||^2)

    This is NOT peak-signal PSNR. It uses the total signal energy
    as the reference, matching the original implementation.

    Parameters
    ----------
    estimate : ndarray
        Reconstructed image.
    reference : ndarray
        Ground truth image.

    Returns
    -------
    psnr : float
        Signal-power PSNR in dB.
    """
    norm_ref = np.sum(np.abs(reference) ** 2)
    norm_err = np.sum(np.abs(estimate - reference) ** 2)
    return float(10 * np.log10(norm_ref / norm_err))


def compute_metrics(estimate: np.ndarray, reference: np.ndarray) -> dict:
    """
    Compute reconstruction quality metrics (benchmark-standard).

    Parameters
    ----------
    estimate : ndarray
        Reconstructed image.
    reference : ndarray
        Ground truth image.

    Returns
    -------
    dict with keys:
        nrmse : float
            Normalized RMS error (dynamic range normalization).
        ncc : float
            Normalized cross-correlation (cosine similarity).
        psnr : float
            Signal-power PSNR in dB.
    """
    est = estimate.flatten().astype(np.float64)
    ref = reference.flatten().astype(np.float64)

    # NRMSE
    mse = np.mean((est - ref) ** 2)
    dynamic_range = ref.max() - ref.min()
    nrmse = np.sqrt(mse) / dynamic_range if dynamic_range > 0 else float("inf")

    # NCC (cosine similarity)
    ncc = np.dot(est, ref) / (np.linalg.norm(est) * np.linalg.norm(ref) + 1e-12)

    # Signal-power PSNR
    psnr = compute_psnr(estimate, reference)

    return {"nrmse": float(nrmse), "ncc": float(ncc), "psnr": float(psnr)}


def plot_reconstruction_comparison(
    recon, zerofill, ground_truth,
    save_path=None,
):
    """
    Plot side-by-side: ground truth, zero-fill, PnP-ADMM reconstruction.

    Parameters
    ----------
    recon : ndarray, (m, n)
        PnP-ADMM reconstruction.
    zerofill : ndarray, (m, n)
        Zero-filled reconstruction.
    ground_truth : ndarray, (m, n)
        Ground truth image.
    save_path : str, optional
        Path to save figure.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    titles = ["Ground Truth", "Zero-Fill", "PnP-ADMM"]
    images = [ground_truth, zerofill, recon]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    m_zf = compute_metrics(zerofill, ground_truth)
    m_pnp = compute_metrics(recon, ground_truth)
    axes[1].set_title(f"Zero-Fill\nPSNR={m_zf['psnr']:.2f} dB")
    axes[2].set_title(f"PnP-ADMM\nPSNR={m_pnp['psnr']:.2f} dB")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_maps(
    recon, zerofill, ground_truth,
    save_path=None,
):
    """
    Plot error maps for zero-fill and PnP-ADMM reconstruction.

    Parameters
    ----------
    recon, zerofill, ground_truth : ndarray, (m, n)
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    err_zf = np.abs(zerofill - ground_truth)
    err_pnp = np.abs(recon - ground_truth)
    vmax = max(err_zf.max(), err_pnp.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    im0 = axes[0].imshow(err_zf, cmap="hot", vmin=0, vmax=vmax)
    axes[0].set_title("Zero-Fill Error")
    axes[0].axis("off")

    im1 = axes[1].imshow(err_pnp, cmap="hot", vmin=0, vmax=vmax)
    axes[1].set_title("PnP-ADMM Error")
    axes[1].axis("off")

    plt.colorbar(im1, ax=axes, shrink=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_psnr_convergence(psnr_history, save_path=None):
    """
    Plot PSNR vs iteration.

    Parameters
    ----------
    psnr_history : ndarray, (maxitr,)
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(psnr_history) + 1), psnr_history, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PnP-ADMM Convergence")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mask(mask, title="Undersampling Mask", save_path=None):
    """Plot a k-space undersampling mask."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
    sampling_pct = mask.sum() / mask.size * 100
    ax.set_title(f"{title} ({sampling_pct:.1f}% sampled)")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_metrics(metrics: dict, label: str = ""):
    """Pretty-print metrics."""
    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}PSNR:  {metrics['psnr']:.2f} dB")
    print(f"{prefix}NCC:   {metrics['ncc']:.4f}")
    print(f"{prefix}NRMSE: {metrics['nrmse']:.6f}")
