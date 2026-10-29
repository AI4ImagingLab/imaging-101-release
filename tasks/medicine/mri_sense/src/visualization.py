"""
Visualization and Metrics for CG-SENSE MRI Reconstruction
============================================================
"""

import numpy as np


def compute_metrics(estimate: np.ndarray, reference: np.ndarray) -> dict:
    """Compute NRMSE, NCC, and SSIM."""
    from skimage.metrics import structural_similarity as _ssim

    est = np.abs(estimate).flatten().astype(np.float64)
    ref = np.abs(reference).flatten().astype(np.float64)

    drange = ref.max() - ref.min()
    nrmse = np.sqrt(np.mean((est - ref) ** 2)) / drange if drange > 0 else float("inf")
    ncc = np.dot(est, ref) / (np.linalg.norm(est) * np.linalg.norm(ref) + 1e-12)

    ref_2d = np.abs(reference).astype(np.float64)
    est_2d = np.abs(estimate).astype(np.float64)
    ssim_val = _ssim(ref_2d, est_2d, data_range=float(drange))

    return {"nrmse": float(nrmse), "ncc": float(ncc), "ssim": float(ssim_val)}


def plot_reconstruction_comparison(recon, zerofill, reference, save_path=None):
    """Plot ground truth, zero-fill, and CG-SENSE side-by-side."""
    import matplotlib.pyplot as plt

    m_s = compute_metrics(recon, reference)
    m_zf = compute_metrics(zerofill, reference)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, img, title in zip(axes,
        [reference, zerofill, recon],
        ["Ground Truth",
         f"Zero-Fill (aliased)\nSSIM={m_zf['ssim']:.3f}",
         f"CG-SENSE\nSSIM={m_s['ssim']:.3f}"]):
        ax.imshow(np.abs(img), cmap="gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_maps(recon, zerofill, reference, save_path=None):
    """Plot error maps."""
    import matplotlib.pyplot as plt

    err_zf = np.abs(np.abs(zerofill) - np.abs(reference))
    err_s = np.abs(np.abs(recon) - np.abs(reference))
    vmax = max(err_zf.max(), err_s.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(err_zf, cmap="hot", vmin=0, vmax=vmax)
    axes[0].set_title("Zero-Fill Error")
    axes[0].axis("off")
    im = axes[1].imshow(err_s, cmap="hot", vmin=0, vmax=vmax)
    axes[1].set_title("CG-SENSE Error")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_maps(sens, save_path=None):
    """Plot coil sensitivity map magnitudes."""
    import matplotlib.pyplot as plt

    nc = sens.shape[-1]
    cols = min(nc, 4)
    rows = (nc + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for i in range(nc):
        r, c = divmod(i, cols)
        axes[r, c].imshow(np.abs(sens[..., i]), cmap="viridis")
        axes[r, c].set_title(f"Coil {i}")
        axes[r, c].axis("off")
    for i in range(nc, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")
    plt.suptitle("Coil Sensitivity Maps (magnitude)")
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
