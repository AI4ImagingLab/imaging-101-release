"""
Visualization and metric utilities for dual-energy CT material decomposition.
"""

import numpy as np


def compute_ncc(estimate, reference):
    """Normalised cross-correlation (cosine similarity) between two arrays.

    Parameters
    ----------
    estimate : ndarray
        Estimated image/map.
    reference : ndarray
        Reference image/map.

    Returns
    -------
    ncc : float
        Cosine similarity in [0, 1] for non-negative signals.
    """
    e = estimate.ravel().astype(np.float64)
    r = reference.ravel().astype(np.float64)
    norm_e = np.linalg.norm(e)
    norm_r = np.linalg.norm(r)
    if norm_e < 1e-30 or norm_r < 1e-30:
        return 0.0
    return float(np.dot(e, r) / (norm_e * norm_r))


def compute_nrmse(estimate, reference):
    """Normalised root-mean-square error.

    NRMSE = RMSE / (max(ref) - min(ref)).

    Parameters
    ----------
    estimate : ndarray
        Estimated image/map.
    reference : ndarray
        Reference image/map.

    Returns
    -------
    nrmse : float
    """
    e = estimate.ravel().astype(np.float64)
    r = reference.ravel().astype(np.float64)
    drange = r.max() - r.min()
    if drange < 1e-30:
        return float("inf")
    rmse = np.sqrt(np.mean((e - r) ** 2))
    return float(rmse / drange)


def compute_metrics(tissue_est, tissue_ref, bone_est, bone_ref):
    """Compute NCC and NRMSE for tissue and bone maps.

    Metrics are computed within a body mask (where tissue_ref + bone_ref > 0).

    Parameters
    ----------
    tissue_est, tissue_ref : ndarray, shape (N, N)
    bone_est, bone_ref : ndarray, shape (N, N)

    Returns
    -------
    metrics : dict
        Keys: tissue_ncc, tissue_nrmse, bone_ncc, bone_nrmse,
              mean_ncc, mean_nrmse.
    """
    # Body mask: any material present
    mask = (tissue_ref + bone_ref) > 0.01

    t_ncc = compute_ncc(tissue_est[mask], tissue_ref[mask])
    t_nrmse = compute_nrmse(tissue_est[mask], tissue_ref[mask])
    b_ncc = compute_ncc(bone_est[mask], bone_ref[mask])
    b_nrmse = compute_nrmse(bone_est[mask], bone_ref[mask])

    return {
        "tissue_ncc": round(t_ncc, 4),
        "tissue_nrmse": round(t_nrmse, 4),
        "bone_ncc": round(b_ncc, 4),
        "bone_nrmse": round(b_nrmse, 4),
        "mean_ncc": round((t_ncc + b_ncc) / 2, 4),
        "mean_nrmse": round((t_nrmse + b_nrmse) / 2, 4),
    }


def plot_material_maps(tissue_est, bone_est, tissue_ref, bone_ref,
                       save_path=None):
    """Plot estimated vs reference material density maps.

    Parameters
    ----------
    tissue_est, bone_est : ndarray, shape (N, N)
        Estimated maps.
    tissue_ref, bone_ref : ndarray, shape (N, N)
        Ground truth maps.
    save_path : str or None
        If provided, save figure to this path.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Tissue row
    im0 = axes[0, 0].imshow(tissue_ref, cmap="gray", vmin=0)
    axes[0, 0].set_title("Tissue (Ground Truth)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(tissue_est, cmap="gray", vmin=0)
    axes[0, 1].set_title("Tissue (Estimated)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    diff_t = tissue_est - tissue_ref
    vabs = max(abs(diff_t.min()), abs(diff_t.max()), 0.01)
    im2 = axes[0, 2].imshow(diff_t, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    axes[0, 2].set_title("Tissue Difference")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Bone row
    im3 = axes[1, 0].imshow(bone_ref, cmap="hot", vmin=0)
    axes[1, 0].set_title("Bone (Ground Truth)")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(bone_est, cmap="hot", vmin=0)
    axes[1, 1].set_title("Bone (Estimated)")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    diff_b = bone_est - bone_ref
    vabs_b = max(abs(diff_b.min()), abs(diff_b.max()), 0.01)
    im5 = axes[1, 2].imshow(diff_b, cmap="RdBu_r", vmin=-vabs_b, vmax=vabs_b)
    axes[1, 2].set_title("Bone Difference")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    for ax in axes.ravel():
        ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sinograms(sino_low, sino_high, save_path=None):
    """Plot low- and high-energy sinograms.

    Parameters
    ----------
    sino_low, sino_high : ndarray, shape (nBins, nAngles)
    save_path : str or None

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(sino_low, aspect="auto", cmap="inferno")
    axes[0].set_title("Low-Energy Sinogram")
    axes[0].set_xlabel("Angle index")
    axes[0].set_ylabel("Detector bin")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(sino_high, aspect="auto", cmap="inferno")
    axes[1].set_title("High-Energy Sinogram")
    axes[1].set_xlabel("Angle index")
    axes[1].set_ylabel("Detector bin")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_spectra_and_mac(energies, spectra, mus, save_path=None):
    """Plot X-ray spectra and mass attenuation coefficients.

    Parameters
    ----------
    energies : ndarray, shape (nE,)
    spectra : ndarray, shape (2, nE)
    mus : ndarray, shape (2, nE)
    save_path : str or None

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(energies, spectra[0], label="Low energy (80 kVp)")
    ax1.plot(energies, spectra[1], label="High energy (140 kVp)")
    ax1.set_xlabel("Energy (keV)")
    ax1.set_ylabel("Photon fluence (per bin)")
    ax1.set_title("X-ray Spectra")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(energies, mus[0], label="Tissue (ICRU)")
    ax2.semilogy(energies, mus[1], label="Bone (ICRU)")
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel("MAC (cm$^2$/g)")
    ax2.set_title("Mass Attenuation Coefficients")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
