"""
Visualization and evaluation metrics for CT reconstruction.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ncc(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Normalized cross-correlation (cosine similarity).

    NCC = (x_hat . x_ref) / (||x_hat|| * ||x_ref||)

    Args:
        estimate: Reconstructed image (any shape, flattened internally).
        reference: Reference image (same shape).

    Returns:
        NCC value in [-1, 1].
    """
    a = estimate.flatten().astype(np.float64)
    b = reference.flatten().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_nrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Normalized root mean squared error.

    NRMSE = sqrt(mean((x_hat - x_ref)^2)) / (max(x_ref) - min(x_ref))

    Args:
        estimate: Reconstructed image.
        reference: Reference image.

    Returns:
        NRMSE value >= 0.
    """
    drange = float(reference.max() - reference.min())
    if drange == 0:
        return 0.0
    rmse = float(np.sqrt(np.mean((estimate - reference) ** 2)))
    return rmse / drange


def centre_crop(image: np.ndarray, fraction: float = 0.8) -> np.ndarray:
    """Extract the central region of a 2D image.

    Args:
        image: 2D array of shape (H, W).
        fraction: Fraction of each dimension to keep (0, 1].

    Returns:
        Cropped 2D array.
    """
    h, w = image.shape
    ch, cw = int(h * fraction), int(w * fraction)
    r0 = (h - ch) // 2
    c0 = (w - cw) // 2
    return image[r0:r0 + ch, c0:c0 + cw]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reconstruction_comparison(phantom, recon_unweighted, recon_pwls,
                                   save_path=None):
    """Plot ground truth, unweighted recon, and PWLS recon side by side.

    Args:
        phantom: 2D ground truth image.
        recon_unweighted: 2D unweighted reconstruction.
        recon_pwls: 2D PWLS reconstruction.
        save_path: If given, save figure to this path.
    """
    import matplotlib.pyplot as plt

    vmin = phantom.min()
    vmax = phantom.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(phantom, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(recon_unweighted, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("Unweighted SVMBIR")
    axes[1].axis("off")

    axes[2].imshow(recon_pwls, cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].set_title("PWLS SVMBIR")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sinogram_comparison(sino_clean, sino_noisy_low, sino_noisy_high,
                              save_path=None):
    """Plot clean vs noisy sinograms at two dose levels.

    Args:
        sino_clean: 2D clean sinogram (V, C).
        sino_noisy_low: 2D low-dose noisy sinogram.
        sino_noisy_high: 2D high-dose noisy sinogram.
        save_path: If given, save figure to this path.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(sino_clean, aspect="auto", cmap="gray")
    axes[0].set_title("Clean Sinogram")
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("View")

    axes[1].imshow(sino_noisy_low, aspect="auto", cmap="gray")
    axes[1].set_title("Low-Dose Sinogram (I0=1000)")
    axes[1].set_xlabel("Channel")

    axes[2].imshow(sino_noisy_high, aspect="auto", cmap="gray")
    axes[2].set_title("High-Dose Sinogram (I0=50000)")
    axes[2].set_xlabel("Channel")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dose_comparison(phantom, recon_low_uw, recon_low_pwls,
                          recon_high_pwls, save_path=None):
    """Plot reconstructions at different dose levels.

    Args:
        phantom: 2D ground truth.
        recon_low_uw: 2D unweighted low-dose recon.
        recon_low_pwls: 2D PWLS low-dose recon.
        recon_high_pwls: 2D PWLS high-dose recon.
        save_path: If given, save figure.
    """
    import matplotlib.pyplot as plt

    vmin = phantom.min()
    vmax = phantom.max()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    titles = ["Ground Truth", "Low-Dose Unweighted",
              "Low-Dose PWLS", "High-Dose PWLS"]
    images = [phantom, recon_low_uw, recon_low_pwls, recon_high_pwls]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_maps(phantom, recon_unweighted, recon_pwls, save_path=None):
    """Plot absolute error maps for unweighted and PWLS reconstructions.

    Args:
        phantom: 2D ground truth.
        recon_unweighted: 2D unweighted reconstruction.
        recon_pwls: 2D PWLS reconstruction.
        save_path: If given, save figure.
    """
    import matplotlib.pyplot as plt

    err_uw = np.abs(recon_unweighted - phantom)
    err_pwls = np.abs(recon_pwls - phantom)
    vmax_err = max(err_uw.max(), err_pwls.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im0 = axes[0].imshow(err_uw, cmap="hot", vmin=0, vmax=vmax_err)
    axes[0].set_title("Error: Unweighted")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(err_pwls, cmap="hot", vmin=0, vmax=vmax_err)
    axes[1].set_title("Error: PWLS")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
