import numpy as np
import matplotlib.pyplot as plt
import os


def compute_snr(ground_truth, estimate):
    """Compute Signal-to-Noise Ratio in dB.

    SNR = 20 * log10(||x_true|| / ||x_true - x_hat||)

    Args:
        ground_truth: ndarray — reference image
        estimate: ndarray — reconstructed image

    Returns:
        float — SNR in dB
    """
    return 20 * np.log10(
        np.linalg.norm(ground_truth.flatten("F"))
        / np.linalg.norm(ground_truth.flatten("F") - estimate.flatten("F"))
    )


def compute_metrics(ground_truth, estimate):
    """Compute reconstruction quality metrics.

    Args:
        ground_truth: ndarray (N, M) — reference image
        estimate: ndarray (N, M) — reconstructed image

    Returns:
        dict with keys: snr_db, nrmse, ncc
    """
    gt = ground_truth.flatten()
    est = estimate.flatten()

    snr_db = compute_snr(ground_truth, estimate)
    nrmse = np.linalg.norm(gt - est) / np.linalg.norm(gt)
    ncc = np.dot(gt, est) / (np.linalg.norm(gt) * np.linalg.norm(est))

    return {"snr_db": snr_db, "nrmse": nrmse, "ncc": ncc}


def plot_comparison(ground_truth, ifft_recon, pnp_recon, metrics, save_path=None):
    """Side-by-side comparison of ground truth, IFFT, and PnP-MSSN.

    Args:
        ground_truth: ndarray (N, M)
        ifft_recon: ndarray (N, M)
        pnp_recon: ndarray (N, M)
        metrics: dict — with keys 'ifft' and 'pnp_mssn', each containing metric dicts
        save_path: str or None
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(ground_truth, cmap="gray")
    axes[0].set_title("Ground Truth", fontsize=14)
    axes[0].axis("off")

    ifft_snr = metrics["ifft"]["snr_db"]
    axes[1].imshow(ifft_recon, cmap="gray")
    axes[1].set_title(f"IFFT (SNR: {ifft_snr:.2f} dB)", fontsize=14)
    axes[1].axis("off")

    pnp_snr = metrics["pnp_mssn"]["snr_db"]
    axes[2].imshow(pnp_recon, cmap="gray")
    axes[2].set_title(f"PnP-MSSN (SNR: {pnp_snr:.2f} dB)", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_convergence(history, save_path=None):
    """Plot SNR and fixed-point residual vs iteration.

    Args:
        history: dict with keys 'snr', 'dist'
        save_path: str or None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if history["snr"]:
        iters = range(1, len(history["snr"]) + 1)
        axes[0].plot(iters, history["snr"], "b-", linewidth=2)
        axes[0].set_xlabel("Iteration", fontsize=13)
        axes[0].set_ylabel("SNR (dB)", fontsize=13)
        axes[0].set_title("SNR vs. Iteration", fontsize=15)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(
            y=history["snr"][-1], color="r", linestyle="--", alpha=0.5,
            label=f"Final: {history['snr'][-1]:.2f} dB",
        )
        axes[0].legend(fontsize=12)

    iters = range(1, len(history["dist"]) + 1)
    axes[1].semilogy(iters, history["dist"], "g-", linewidth=2)
    axes[1].set_xlabel("Iteration", fontsize=13)
    axes[1].set_ylabel("||x - Prox(x)||²", fontsize=13)
    axes[1].set_title("Fixed-Point Residual", fontsize=15)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_maps(ground_truth, ifft_recon, pnp_recon, mask, save_path=None):
    """Error maps for IFFT and PnP-MSSN, plus sampling mask.

    Args:
        ground_truth: ndarray (N, M)
        ifft_recon: ndarray (N, M)
        pnp_recon: ndarray (N, M)
        mask: ndarray (N, M) — sampling mask
        save_path: str or None
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ifft_error = np.abs(ground_truth - ifft_recon)
    ifft_snr = compute_snr(ground_truth, ifft_recon)
    im0 = axes[0].imshow(ifft_error, cmap="hot", vmin=0, vmax=0.3)
    axes[0].set_title(f"IFFT Error (SNR: {ifft_snr:.2f} dB)", fontsize=14)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    pnp_error = np.abs(ground_truth - pnp_recon)
    pnp_snr = compute_snr(ground_truth, pnp_recon)
    im1 = axes[1].imshow(pnp_error, cmap="hot", vmin=0, vmax=0.3)
    axes[1].set_title(f"PnP-MSSN Error (SNR: {pnp_snr:.2f} dB)", fontsize=14)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Sampling Mask ({int(mask.sum())} samples)", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_progression(recon_dict, ground_truth, save_path=None):
    """Show reconstruction at selected iterations.

    Args:
        recon_dict: dict mapping iteration number -> ndarray (N, M)
        ground_truth: ndarray (N, M)
        save_path: str or None
    """
    iters = sorted(recon_dict.keys())
    n = len(iters)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, it in enumerate(iters):
        recon = recon_dict[it]
        snr = compute_snr(ground_truth, recon)
        axes[idx].imshow(recon, cmap="gray")
        axes[idx].set_title(f"Iter {it}\n{snr:.1f} dB", fontsize=11)
        axes[idx].axis("off")

    plt.suptitle("PnP-MSSN Reconstruction Progression", fontsize=16, y=1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_metrics_table(metrics):
    """Print a formatted metrics comparison table.

    Args:
        metrics: dict mapping method name -> metric dict
    """
    header = f"{'Method':<20} {'SNR (dB)':>10} {'NRMSE':>10} {'NCC':>10}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(f"{name:<20} {m['snr_db']:>10.2f} {m['nrmse']:>10.4f} {m['ncc']:>10.4f}")
