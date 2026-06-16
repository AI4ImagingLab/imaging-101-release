"""
Visualization and Metrics for GRAPPA MRI Reconstruction
=========================================================
"""

import numpy as np


def compute_metrics(estimate: np.ndarray, reference: np.ndarray) -> dict:
    """
    Compute NRMSE, NCC, and SSIM.

    Parameters
    ----------
    estimate, reference : ndarray
        Images to compare.

    Returns
    -------
    dict with nrmse, ncc, ssim keys.
    """
    from skimage.metrics import structural_similarity as _ssim

    est = estimate.flatten().astype(np.float64)
    ref = reference.flatten().astype(np.float64)

    drange = ref.max() - ref.min()
    nrmse = np.sqrt(np.mean((est - ref) ** 2)) / drange if drange > 0 else float("inf")
    ncc = np.dot(est, ref) / (np.linalg.norm(est) * np.linalg.norm(ref) + 1e-12)
    ssim_val = _ssim(reference, estimate, data_range=float(drange))

    return {"nrmse": float(nrmse), "ncc": float(ncc), "ssim": float(ssim_val)}


def plot_reconstruction_comparison(recon, zerofill, reference, save_path=None):
    """Plot ground truth, zero-fill, and GRAPPA side-by-side."""
    import matplotlib.pyplot as plt

    m_g = compute_metrics(recon, reference)
    m_zf = compute_metrics(zerofill, reference)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, img, title in zip(axes,
        [reference, zerofill, recon],
        ["Fully Sampled", f"Zero-Fill\nSSIM={m_zf['ssim']:.3f}", f"GRAPPA\nSSIM={m_g['ssim']:.3f}"]):
        ax.imshow(np.abs(img), cmap="gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_maps(recon, zerofill, reference, save_path=None):
    """Plot error maps for zero-fill and GRAPPA."""
    import matplotlib.pyplot as plt

    err_zf = np.abs(zerofill - reference)
    err_g = np.abs(recon - reference)
    vmax = max(err_zf.max(), err_g.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(err_zf, cmap="hot", vmin=0, vmax=vmax)
    axes[0].set_title("Zero-Fill Error")
    axes[0].axis("off")
    im = axes[1].imshow(err_g, cmap="hot", vmin=0, vmax=vmax)
    axes[1].set_title("GRAPPA Error")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kspace(kspace_us, kspace_recon, save_path=None):
    """Plot undersampled and GRAPPA-reconstructed k-space (log magnitude)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, ks, title in zip(axes,
        [kspace_us[..., 0], kspace_recon[..., 0]],
        ["Undersampled (coil 0)", "GRAPPA Reconstructed (coil 0)"]):
        ax.imshow(np.log1p(np.abs(ks)), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_metrics(metrics: dict, label: str = ""):
    """Pretty-print metrics."""
    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}SSIM:  {metrics['ssim']:.4f}")
    print(f"{prefix}NCC:   {metrics['ncc']:.6f}")
    print(f"{prefix}NRMSE: {metrics['nrmse']:.6f}")
