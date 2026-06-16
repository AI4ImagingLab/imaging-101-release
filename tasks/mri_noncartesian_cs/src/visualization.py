"""
Visualization and Metrics for Non-Cartesian MRI Reconstruction
================================================================

Provides metrics computation (NRMSE, NCC, PSNR) and plotting utilities
for non-Cartesian MRI reconstruction results.
"""

import numpy as np


def compute_metrics(estimate: np.ndarray, reference: np.ndarray) -> dict:
    """
    Compute reconstruction quality metrics between magnitude images.

    Parameters
    ----------
    estimate : ndarray, (H, W) or (N, H, W)
        Reconstructed magnitude image(s).
    reference : ndarray, (H, W) or (N, H, W)
        Reference magnitude image(s).

    Returns
    -------
    dict with keys:
        nrmse : float
            Normalized RMS error (normalized by dynamic range of reference).
        ncc : float
            Normalized cross-correlation (cosine similarity).
        psnr : float
            Peak signal-to-noise ratio (dB).
    """
    est = estimate.flatten().astype(np.float64)
    ref = reference.flatten().astype(np.float64)

    mse = np.mean((est - ref) ** 2)
    dynamic_range = ref.max() - ref.min()
    nrmse = np.sqrt(mse) / dynamic_range if dynamic_range > 0 else float("inf")

    ncc = np.dot(est, ref) / (np.linalg.norm(est) * np.linalg.norm(ref) + 1e-12)

    data_range = ref.max()
    psnr = 10.0 * np.log10(data_range ** 2 / mse) if mse > 0 else float("inf")

    return {"nrmse": float(nrmse), "ncc": float(ncc), "psnr": float(psnr)}


def compute_batch_metrics(
    estimates: np.ndarray, references: np.ndarray
) -> dict:
    """
    Compute per-sample and average metrics for a batch.

    Parameters
    ----------
    estimates : ndarray, (N, H, W)
        Batch of reconstructed magnitude images.
    references : ndarray, (N, H, W)
        Batch of reference magnitude images.

    Returns
    -------
    dict with keys:
        per_sample : list of dict
        avg_nrmse, avg_ncc, avg_psnr : float
    """
    per_sample = []
    for i in range(len(estimates)):
        m = compute_metrics(estimates[i], references[i])
        per_sample.append(m)

    return {
        "per_sample": per_sample,
        "avg_nrmse": float(np.mean([m["nrmse"] for m in per_sample])),
        "avg_ncc": float(np.mean([m["ncc"] for m in per_sample])),
        "avg_psnr": float(np.mean([m["psnr"] for m in per_sample])),
    }


def plot_reconstruction_comparison(
    ground_truth,
    gridding,
    l1wav,
    save_path=None,
):
    """
    Plot side-by-side comparison of ground truth, gridding, and L1-wavelet
    reconstructions.

    Parameters
    ----------
    ground_truth : ndarray, (H, W)
        Ground truth magnitude image.
    gridding : ndarray, (H, W)
        Gridding reconstruction magnitude.
    l1wav : ndarray, (H, W)
        L1-wavelet reconstruction magnitude.
    save_path : str, optional
        Path to save the figure.
    """
    import matplotlib.pyplot as plt

    vmax = ground_truth.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ground_truth, cmap="gray", vmin=0, vmax=vmax)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    m_grid = compute_metrics(gridding, ground_truth)
    axes[1].imshow(gridding, cmap="gray", vmin=0, vmax=vmax)
    axes[1].set_title(
        f"Gridding\nPSNR={m_grid['psnr']:.1f}dB, NCC={m_grid['ncc']:.3f}"
    )
    axes[1].axis("off")

    m_l1 = compute_metrics(l1wav, ground_truth)
    axes[2].imshow(l1wav, cmap="gray", vmin=0, vmax=vmax)
    axes[2].set_title(
        f"L1-Wavelet CS\nPSNR={m_l1['psnr']:.1f}dB, NCC={m_l1['ncc']:.3f}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_maps(
    ground_truth,
    gridding,
    l1wav,
    save_path=None,
):
    """
    Plot error maps for gridding and L1-wavelet reconstructions.

    Parameters
    ----------
    ground_truth : ndarray, (H, W)
    gridding : ndarray, (H, W)
    l1wav : ndarray, (H, W)
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    err_grid = np.abs(ground_truth - gridding)
    err_l1 = np.abs(ground_truth - l1wav)
    vmax = max(err_grid.max(), err_l1.max())

    im0 = axes[0].imshow(err_grid, cmap="hot", vmin=0, vmax=vmax)
    axes[0].set_title("Gridding Error")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(err_l1, cmap="hot", vmin=0, vmax=vmax)
    axes[1].set_title("L1-Wavelet Error")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(coord, n_spokes=None, save_path=None):
    """
    Visualize the radial k-space trajectory.

    Parameters
    ----------
    coord : ndarray, (M, 2) float
        k-space coordinates.
    n_spokes : int, optional
        Number of spokes (for coloring). If None, all points are one color.
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if n_spokes is not None:
        n_readout = coord.shape[0] // n_spokes
        for i in range(n_spokes):
            start = i * n_readout
            end = start + n_readout
            ax.plot(
                coord[start:end, 0],
                coord[start:end, 1],
                linewidth=0.5,
                alpha=0.6,
            )
    else:
        ax.scatter(coord[:, 0], coord[:, 1], s=0.1, alpha=0.5)

    ax.set_aspect("equal")
    ax.set_title("Radial k-space Trajectory")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_metrics_table(metrics: dict):
    """
    Pretty-print metrics for reconstruction methods.

    Parameters
    ----------
    metrics : dict
        Dict mapping method name to metrics dict.
    """
    print(f"{'Method':>20s}  {'PSNR (dB)':>10s}  {'NCC':>8s}  {'NRMSE':>8s}")
    print("-" * 52)
    for name, m in metrics.items():
        print(f"{name:>20s}  {m['psnr']:>10.2f}  {m['ncc']:>8.4f}  {m['nrmse']:>8.4f}")
