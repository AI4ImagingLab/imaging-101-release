"""
Visualisation utilities for lensless imaging results.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
    mean_squared_error as mse,
)


def normalise_for_display(img: np.ndarray) -> np.ndarray:
    """Shift and scale an image to [0, 1] for display."""
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img


def gamma_correction(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Apply gamma correction: img^(1/gamma), after clipping to [0, 1]."""
    img = np.clip(img, 0, 1)
    return img ** (1.0 / gamma)


def plot_overview(
    psf: np.ndarray,
    measurement: np.ndarray,
    reconstruction: np.ndarray,
    gamma: float = 2.2,
    save_path: str | None = None,
):
    """Three-panel figure: PSF | raw measurement | reconstruction.

    Parameters
    ----------
    psf            : ndarray (H, W, C)
    measurement    : ndarray (H, W, C)
    reconstruction : ndarray (H, W, C)
    gamma          : float  Display gamma.
    save_path      : str or None  If given, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    panels = [
        (psf,            "PSF (gamma corrected)"),
        (measurement,    "Raw lensless measurement"),
        (reconstruction, "ADMM reconstruction"),
    ]

    for ax, (img, title) in zip(axes, panels):
        disp = gamma_correction(normalise_for_display(img), gamma)
        ax.imshow(disp)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_convergence(residuals: list, save_path: str | None = None):
    """Plot reconstruction residual vs iteration.

    Parameters
    ----------
    residuals : list of float  Residual value at each iteration.
    save_path : optional str
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(residuals)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("||Av - b||² / ||b||²")
    ax.set_title("ADMM convergence")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compute_metrics(reconstruction: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Compute image quality metrics between reconstruction and ground truth.

    Parameters
    ----------
    reconstruction : ndarray (H, W, C)  Values in [0, 1].
    ground_truth   : ndarray (H, W, C)  Values in [0, 1].

    Returns
    -------
    dict with keys: mse, psnr, ssim
    """
    rec  = normalise_for_display(reconstruction)
    gt   = normalise_for_display(ground_truth)
    rec  = np.clip(rec, 0, 1)
    gt   = np.clip(gt,  0, 1)

    mse_val  = float(mse(gt, rec))
    psnr_val = float(psnr(gt, rec, data_range=1.0))
    # SSIM requires channel_axis for colour images
    ssim_val = float(ssim(gt, rec, data_range=1.0, channel_axis=-1))

    return {"mse": mse_val, "psnr": psnr_val, "ssim": ssim_val}
