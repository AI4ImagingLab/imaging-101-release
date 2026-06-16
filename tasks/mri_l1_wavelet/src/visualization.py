"""
Visualization and Metrics for MRI L1-Wavelet Reconstruction
=============================================================

Provides metrics computation (NRMSE, NCC, PSNR) and plotting utilities
for multi-coil MRI reconstruction results.
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
            Peak signal-to-noise ratio (dB), using max of reference as data range.
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
            Per-sample metrics.
        avg_nrmse, avg_ncc, avg_psnr : float
            Average metrics across samples.
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


def plot_reconstruction_grid(
    ground_truths,
    reconstructions,
    zero_filled=None,
    tv_recons=None,
    sample_indices=None,
    save_path=None,
):
    """
    Plot a grid comparing ground truth, zero-filled, TV, and L1-wavelet
    reconstructions.

    Parameters
    ----------
    ground_truths : ndarray, (N, H, W)
        Ground truth magnitude images.
    reconstructions : ndarray, (N, H, W)
        L1-Wavelet reconstruction magnitude images.
    zero_filled : ndarray, (N, H, W), optional
        Zero-filled reconstruction magnitude images.
    tv_recons : ndarray, (N, H, W), optional
        TV reconstruction magnitude images.
    sample_indices : list of int, optional
        Which samples to show. Default: all.
    save_path : str, optional
        Path to save the figure.
    """
    import matplotlib.pyplot as plt

    if sample_indices is None:
        sample_indices = list(range(len(ground_truths)))

    n_samples = len(sample_indices)
    n_cols = 2
    if zero_filled is not None:
        n_cols += 1
    if tv_recons is not None:
        n_cols += 1

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 4 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    for row, idx in enumerate(sample_indices):
        gt = ground_truths[idx]
        recon = reconstructions[idx]
        vmax = gt.max()
        col = 0

        axes[row, col].imshow(gt, cmap="gray", vmin=0, vmax=vmax)
        axes[row, col].set_title(f"Ground Truth (#{idx})")
        axes[row, col].axis("off")
        col += 1

        if zero_filled is not None:
            axes[row, col].imshow(zero_filled[idx], cmap="gray", vmin=0, vmax=vmax)
            axes[row, col].set_title(f"Zero-Filled (#{idx})")
            axes[row, col].axis("off")
            col += 1

        if tv_recons is not None:
            axes[row, col].imshow(tv_recons[idx], cmap="gray", vmin=0, vmax=vmax)
            m_tv = compute_metrics(tv_recons[idx], gt)
            axes[row, col].set_title(
                f"TV (#{idx})\nPSNR={m_tv['psnr']:.1f}, NCC={m_tv['ncc']:.3f}"
            )
            axes[row, col].axis("off")
            col += 1

        axes[row, col].imshow(recon, cmap="gray", vmin=0, vmax=vmax)
        m = compute_metrics(recon, gt)
        axes[row, col].set_title(
            f"L1-Wavelet (#{idx})\nPSNR={m['psnr']:.1f}, NCC={m['ncc']:.3f}"
        )
        axes[row, col].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_maps(
    ground_truths,
    reconstructions,
    tv_recons=None,
    sample_indices=None,
    save_path=None,
):
    """
    Plot error maps (absolute difference) between GT and reconstruction.

    Parameters
    ----------
    ground_truths : ndarray, (N, H, W)
        Ground truth magnitude images.
    reconstructions : ndarray, (N, H, W)
        L1-Wavelet reconstructed magnitude images.
    tv_recons : ndarray, (N, H, W), optional
        TV reconstructed magnitude images for comparison.
    sample_indices : list of int, optional
        Which samples to show.
    save_path : str, optional
        Path to save the figure.
    """
    import matplotlib.pyplot as plt

    if sample_indices is None:
        sample_indices = list(range(len(ground_truths)))

    n = len(sample_indices)
    n_cols = 1 if tv_recons is None else 2
    fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 4 * n))
    if n == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]

    for i, idx in enumerate(sample_indices):
        error_wav = np.abs(ground_truths[idx] - reconstructions[idx])
        im = axes[i, 0].imshow(error_wav, cmap="hot")
        axes[i, 0].set_title(f"L1-Wavelet Error (#{idx})")
        axes[i, 0].axis("off")
        plt.colorbar(im, ax=axes[i, 0], fraction=0.046, pad=0.04)

        if tv_recons is not None:
            error_tv = np.abs(ground_truths[idx] - tv_recons[idx])
            im2 = axes[i, 1].imshow(error_tv, cmap="hot")
            axes[i, 1].set_title(f"TV Error (#{idx})")
            axes[i, 1].axis("off")
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_undersampling_mask(mask, save_path=None):
    """
    Visualize the 1-D undersampling mask as a 2-D k-space pattern.

    Parameters
    ----------
    mask : ndarray, (W,)
        1-D binary undersampling mask.
    save_path : str, optional
        Path to save the figure.
    """
    import matplotlib.pyplot as plt

    H = mask.shape[0]
    mask_2d = np.tile(mask[None, :], (H, 1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(mask_2d, cmap="gray", aspect="auto")
    axes[0].set_title(
        f"k-space Sampling Pattern\n"
        f"{int(mask.sum())}/{len(mask)} lines "
        f"({mask.sum() / len(mask) * 100:.1f}%)"
    )
    axes[0].set_xlabel("Phase Encode (kx)")
    axes[0].set_ylabel("Readout (ky)")

    axes[1].stem(mask, markerfmt=".", linefmt="b-", basefmt="r-")
    axes[1].set_title("Sampled Lines")
    axes[1].set_xlabel("Phase Encode Index")
    axes[1].set_ylabel("Sampled")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_metrics_table(batch_metrics: dict, method_name: str = "Recon"):
    """
    Pretty-print per-sample and average metrics.

    Parameters
    ----------
    batch_metrics : dict
        Output from compute_batch_metrics.
    method_name : str
        Label for this reconstruction method.
    """
    print(f"\n  {method_name}:")
    print(f"  {'Sample':>8s}  {'PSNR (dB)':>10s}  {'NCC':>8s}  {'NRMSE':>8s}")
    print("  " + "-" * 40)
    for i, m in enumerate(batch_metrics["per_sample"]):
        print(f"  {i:>8d}  {m['psnr']:>10.2f}  {m['ncc']:>8.4f}  {m['nrmse']:>8.4f}")
    print("  " + "-" * 40)
    print(
        f"  {'Average':>8s}  {batch_metrics['avg_psnr']:>10.2f}  "
        f"{batch_metrics['avg_ncc']:>8.4f}  {batch_metrics['avg_nrmse']:>8.4f}"
    )
