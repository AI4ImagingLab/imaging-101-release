"""
Visualization and Metrics for VarNet MRI Reconstruction
=========================================================
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


def plot_reconstruction_comparison(recons, zerofills, ground_truths,
                                    slice_idx=0, save_path=None):
    """Plot GT, zero-fill, VarNet for a single slice."""
    import matplotlib.pyplot as plt

    gt = ground_truths[slice_idx]
    zf = zerofills[slice_idx]
    vn = recons[slice_idx]

    m_vn = compute_metrics(vn, gt)
    m_zf = compute_metrics(zf, gt)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes,
        [gt, zf, vn],
        ["Ground Truth (RSS)",
         f"Zero-Fill\nSSIM={m_zf['ssim']:.3f}",
         f"VarNet\nSSIM={m_vn['ssim']:.3f}"]):
        ax.imshow(np.abs(img), cmap="gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_maps(recon, zerofill, ground_truth, save_path=None):
    """Plot error maps."""
    import matplotlib.pyplot as plt

    err_zf = np.abs(np.abs(zerofill) - np.abs(ground_truth))
    err_vn = np.abs(np.abs(recon) - np.abs(ground_truth))
    vmax = max(err_zf.max(), err_vn.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(err_zf, cmap="hot", vmin=0, vmax=vmax)
    axes[0].set_title("Zero-Fill Error")
    axes[0].axis("off")
    im = axes[1].imshow(err_vn, cmap="hot", vmin=0, vmax=vmax)
    axes[1].set_title("VarNet Error")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_metrics(metrics: dict, label: str = ""):
    """Pretty-print metrics."""
    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}SSIM:  {metrics['ssim']:.4f}")
    print(f"{prefix}NCC:   {metrics['ncc']:.4f}")
    print(f"{prefix}NRMSE: {metrics['nrmse']:.6f}")
