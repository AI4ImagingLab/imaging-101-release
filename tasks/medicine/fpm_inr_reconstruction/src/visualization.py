"""
Visualization, Metrics, and All-in-Focus for FPM-INR
=====================================================

Plotting utilities for amplitude/phase images, per-slice metrics,
ground truth comparisons, all-in-focus evaluation, and the Normal Variance
focus-stacking method.

IMPORTANT: This module does NOT set matplotlib backend.
Only main.py should call matplotlib.use('Agg').
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ──────────────────────────────────────────────────────────────────────────────
# All-in-Focus Computation
# ──────────────────────────────────────────────────────────────────────────────


def create_balance_map(image_size: int, patch_size: int = 64,
                       patch_pace: int = 16) -> tuple:
    """
    Create balance map for overlapping patch fusion.

    Parameters
    ----------
    image_size : int
        Side length of the square image.
    patch_size : int
        Side length of each patch.
    patch_pace : int
        Stride between patches.

    Returns
    -------
    n_patches : int
        Number of patches along each axis.
    balance_map : ndarray (image_size, image_size)
        Inverse overlap count for weighted fusion.
    """
    n_patches = int((image_size - patch_size) / patch_pace + 1)
    balance_map = np.zeros((image_size, image_size))

    for i in range(n_patches):
        for j in range(n_patches):
            start_ud = i * patch_pace
            end_ud = start_ud + patch_size
            start_lr = j * patch_pace
            end_lr = start_lr + patch_size
            balance_map[start_ud:end_ud, start_lr:end_lr] += np.ones(
                (patch_size, patch_size)
            )

    balance_map = 1 / balance_map
    return n_patches, balance_map


def all_in_focus_normal_variance(z_stack: np.ndarray, patch_size: int = 64,
                                  patch_pace: int = 16) -> np.ndarray:
    """
    All-in-focus using Normal Variance focus measure.

    For each overlapping patch, computes normalized variance across all z-planes
    and selects the plane with maximum sharpness. Overlapping patches are fused
    using the balance map.

    Note: This function makes a copy of the input to avoid mutation.

    Parameters
    ----------
    z_stack : ndarray (H, W, n_z)
        Amplitude images at different z-planes.
    patch_size : int
        Side length of each patch.
    patch_pace : int
        Stride between patches.

    Returns
    -------
    aif_image : ndarray (H, W)
        All-in-focus composite image.
    """
    imgs = z_stack.copy()
    m, n, framenum = imgs.shape

    o_sum = np.mean(imgs, axis=2)
    intensity, edges = np.histogram(np.abs(o_sum), bins="auto")
    pos_i = np.argmax(intensity)
    background = edges[pos_i]

    o_fusion = np.zeros((m, n))
    n_patches, balance_map = create_balance_map(m, patch_size, patch_pace)

    for i in range(n_patches):
        for j in range(n_patches):
            NV = []
            for k in range(framenum):
                start_ud = i * patch_pace
                end_ud = start_ud + patch_size
                start_lr = j * patch_pace
                end_lr = start_lr + patch_size

                o_cropped = imgs[start_ud:end_ud, start_lr:end_lr, k]

                # Rule out FPM artifacts
                if background > 0.1:
                    o_cropped[o_cropped > background] = background

                imgs[start_ud:end_ud, start_lr:end_lr, k] = o_cropped

                mu = np.mean(o_cropped)
                NV.append(
                    np.sum((o_cropped - mu) ** 2) / (patch_size**2 * mu)
                )

            loc_info = np.argmax(NV)
            o_fusion[start_ud:end_ud, start_lr:end_lr] += imgs[
                start_ud:end_ud, start_lr:end_lr, loc_info
            ]

    o_fusion_balanced = o_fusion * balance_map
    return o_fusion_balanced


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────


def compute_metrics(pred_stack: np.ndarray, gt_stack: np.ndarray) -> dict:
    """
    Compute per-slice and overall reconstruction quality metrics.

    Both stacks are normalized to [0, 1] before comparison.

    Parameters
    ----------
    pred_stack : ndarray (n_z, H, W)
        Predicted amplitude images.
    gt_stack : ndarray (n_z, H, W)
        Ground truth amplitude images.

    Returns
    -------
    dict with keys:
        'pred_norm'       : ndarray - normalized predictions
        'gt_norm'         : ndarray - normalized ground truth
        'l2_per_slice'    : ndarray (n_z,) - RMSE per slice
        'mse_per_slice'   : ndarray (n_z,) - MSE per slice
        'psnr_per_slice'  : ndarray (n_z,) - PSNR per slice
        'l2_overall'      : float - overall RMSE
        'mse_overall'     : float - overall MSE
        'psnr_overall'    : float - overall PSNR in dB
    """
    pred_norm = (pred_stack - pred_stack.min()) / (
        pred_stack.max() - pred_stack.min() + 1e-8
    )
    gt_norm = (gt_stack - gt_stack.min()) / (gt_stack.max() - gt_stack.min() + 1e-8)

    mse_per_slice = np.mean((pred_norm - gt_norm) ** 2, axis=(1, 2))
    l2_per_slice = np.sqrt(mse_per_slice)
    psnr_per_slice = 10 * np.log10(1.0 / (mse_per_slice + 1e-10))

    mse_overall = np.mean((pred_norm - gt_norm) ** 2)
    l2_overall = np.sqrt(mse_overall)
    psnr_overall = 10 * np.log10(1.0 / (mse_overall + 1e-10))

    return {
        "pred_norm": pred_norm,
        "gt_norm": gt_norm,
        "l2_per_slice": l2_per_slice,
        "mse_per_slice": mse_per_slice,
        "psnr_per_slice": psnr_per_slice,
        "l2_overall": l2_overall,
        "mse_overall": mse_overall,
        "psnr_overall": psnr_overall,
    }


def compute_ssim_per_slice(pred_norm: np.ndarray, gt_norm: np.ndarray) -> np.ndarray:
    """
    Compute SSIM for each z-slice.

    Parameters
    ----------
    pred_norm : ndarray (n_z, H, W) - normalized predictions [0, 1]
    gt_norm : ndarray (n_z, H, W) - normalized ground truth [0, 1]

    Returns
    -------
    ssim_per_slice : ndarray (n_z,)
    """
    from skimage.metrics import structural_similarity as ssim

    return np.array(
        [
            ssim(pred_norm[i], gt_norm[i], data_range=1.0)
            for i in range(len(pred_norm))
        ]
    )


def compute_allfocus_l2(aif_pred: np.ndarray, aif_gt: np.ndarray) -> dict:
    """
    Compute mean-subtracted L2 error as in paper (Fig. 2 metric).

    Parameters
    ----------
    aif_pred : ndarray (H, W) - predicted all-in-focus image
    aif_gt : ndarray (H, W) - ground truth all-in-focus image

    Returns
    -------
    dict with 'mse' and 'psnr' keys
    """
    pred_centered = aif_pred - np.mean(aif_pred)
    gt_centered = aif_gt - np.mean(aif_gt)
    mse = np.mean((pred_centered - gt_centered) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))

    return {"mse": float(mse), "psnr": float(psnr)}


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────


def plot_amplitude_phase(amplitude: np.ndarray, phase: np.ndarray,
                         epoch: int = None, save_path: str = None,
                         figsize=(20, 10)):
    """
    Plot amplitude and phase side by side with colorbars.

    Parameters
    ----------
    amplitude : ndarray (H, W)
    phase : ndarray (H, W)
    epoch : int, optional - epoch number for title
    save_path : str, optional - path to save figure
    figsize : tuple
    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    im = axs[0].imshow(amplitude, cmap="gray")
    axs[0].axis("image")
    title_suffix = f" (epoch {epoch})" if epoch is not None else ""
    axs[0].set_title(f"Reconstructed amplitude{title_suffix}")
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    im = axs[1].imshow(phase, cmap="gray")
    axs[1].axis("image")
    axs[1].set_title(f"Reconstructed phase{title_suffix}")
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    return fig


def plot_per_slice_metrics(z_positions: np.ndarray, l2: np.ndarray,
                           psnr: np.ndarray, ssim: np.ndarray = None,
                           save_path: str = None, figsize=(20, 5)):
    """
    Plot per-slice L2, PSNR, and optionally SSIM vs z-position.
    """
    ncols = 3 if ssim is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    axes[0].plot(z_positions, l2)
    axes[0].set_xlabel("z (um)")
    axes[0].set_ylabel("L2 (RMSE)")
    axes[0].set_title("Per-slice L2 error")
    axes[0].grid(True)

    axes[1].plot(z_positions, psnr)
    axes[1].set_xlabel("z (um)")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Per-slice PSNR")
    axes[1].grid(True)

    if ssim is not None:
        axes[2].plot(z_positions, ssim)
        axes[2].set_xlabel("z (um)")
        axes[2].set_ylabel("SSIM")
        axes[2].set_title("Per-slice SSIM")
        axes[2].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    return fig


def plot_gt_comparison(pred_norm: np.ndarray, gt_norm: np.ndarray,
                       z_positions: np.ndarray, psnr_per_slice: np.ndarray,
                       l2_per_slice: np.ndarray, save_path: str = None,
                       figsize=(18, 18)):
    """
    Visual comparison at best/middle/worst z-slices.
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    indices = [
        np.argmax(psnr_per_slice),
        len(z_positions) // 2,
        np.argmin(psnr_per_slice),
    ]
    labels = ["Best PSNR", "Middle z", "Worst PSNR"]

    for row, (idx, label) in enumerate(zip(indices, labels)):
        z_val = z_positions[idx]
        im0 = axes[row, 0].imshow(pred_norm[idx], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Predicted (z={z_val:.1f}um) - {label}")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        im1 = axes[row, 1].imshow(gt_norm[idx], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"GT (z={z_val:.1f}um)")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        diff = np.abs(pred_norm[idx] - gt_norm[idx])
        im2 = axes[row, 2].imshow(diff, cmap="hot", vmin=0, vmax=diff.max())
        axes[row, 2].set_title(
            f"|Diff| L2={l2_per_slice[idx]:.4f}, PSNR={psnr_per_slice[idx]:.1f}dB"
        )
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    return fig


def plot_allfocus_comparison(aif_pred: np.ndarray, aif_gt: np.ndarray,
                              l2_error: float, save_path: str = None,
                              figsize=(18, 6)):
    """
    Side-by-side all-in-focus comparison with error map.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.set_dpi(150)

    axes[0].imshow(aif_gt, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("GT All-in-Focus")
    axes[0].axis("off")

    axes[1].imshow(aif_pred, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"FPM-INR All-in-Focus\nL2={l2_error:.4e}")
    axes[1].axis("off")

    error_map = np.abs(
        (aif_pred - np.mean(aif_pred)) - (aif_gt - np.mean(aif_gt))
    )
    im = axes[2].imshow(error_map, cmap="hot", vmin=0, vmax=0.02)
    axes[2].set_title("L2 Error Map")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    return fig
