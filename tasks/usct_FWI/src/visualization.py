"""Plotting utilities and metrics for USCT FWI results."""

import numpy as np
from typing import Optional, List, Tuple


def compute_ncc(x: np.ndarray, ref: np.ndarray) -> float:
    """Normalized Cross-Correlation (cosine similarity)."""
    x_flat = x.flatten().astype(np.float64)
    ref_flat = ref.flatten().astype(np.float64)
    return float(np.dot(x_flat, ref_flat) / (np.linalg.norm(x_flat) * np.linalg.norm(ref_flat)))


def compute_nrmse(x: np.ndarray, ref: np.ndarray) -> float:
    """NRMSE normalized by dynamic range of reference."""
    x_flat = x.flatten().astype(np.float64)
    ref_flat = ref.flatten().astype(np.float64)
    rmse = np.sqrt(np.mean((x_flat - ref_flat) ** 2))
    drange = ref_flat.max() - ref_flat.min()
    return float(rmse / drange) if drange > 0 else float("inf")


def plot_velocity(vp, domain_size_cm=(24, 24), title="", cmap="gray",
                  vmin=None, vmax=None, save_path=None):
    """Plot velocity model with physical axis labels."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    nx, ny = vp.shape
    extent = [0, domain_size_cm[0], 0, domain_size_cm[1]]
    im = ax.imshow(vp, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    ax.set_xlabel("X / cm")
    ax.set_ylabel("Y / cm")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Sound speed (m/s)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_comparison(vp_ref, vp_recon, domain_size_cm=(24, 24),
                    vmin=None, vmax=None, save_path=None):
    """Plot reference vs reconstruction side by side."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    extent = [0, domain_size_cm[0], 0, domain_size_cm[1]]

    for ax, data, title in zip(
        axes,
        [vp_ref, vp_recon],
        ["Reference", "Reconstruction"],
    ):
        im = ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, extent=extent)
        ax.set_xlabel("X / cm")
        ax.set_ylabel("Y / cm")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Sound speed (m/s)")

    ncc = compute_ncc(vp_recon, vp_ref)
    nrmse = compute_nrmse(vp_recon, vp_ref)
    fig.suptitle(f"NCC={ncc:.4f}, NRMSE={nrmse:.4f}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_dobs(dobs, freq, save_path=None):
    """Plot observation data magnitude."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.abs(dobs), origin="lower")
    ax.set_title(f"|dobs| at {freq} MHz")
    ax.set_xlabel("Source index")
    ax.set_ylabel("Receiver index")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_convergence(history: List[Tuple[float, np.ndarray]],
                     vp_ref: np.ndarray = None, save_path=None):
    """Plot NCC/NRMSE vs frequency for multi-frequency history."""
    import matplotlib.pyplot as plt
    if vp_ref is None:
        return None
    freqs = [h[0] for h in history]
    nccs = [compute_ncc(h[1], vp_ref) for h in history]
    nrmses = [compute_nrmse(h[1], vp_ref) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(freqs, nccs, "o-")
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("NCC")
    ax1.set_title("NCC vs Frequency")
    ax1.grid(True)

    ax2.plot(freqs, nrmses, "o-")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("NRMSE")
    ax2.set_title("NRMSE vs Frequency")
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
