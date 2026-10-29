"""Visualization utilities and evaluation metrics."""

import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(estimate, reference):
    """Compute NCC and NRMSE between estimate and reference arrays.

    NCC is the cosine similarity (no mean subtraction).
    NRMSE is RMS error normalised by the dynamic range of the reference.

    Parameters
    ----------
    estimate : np.ndarray
    reference : np.ndarray

    Returns
    -------
    dict with keys 'ncc' and 'nrmse'.
    """
    est = estimate.ravel().astype(np.float64)
    ref = reference.ravel().astype(np.float64)

    # NCC: cosine similarity
    ncc = np.dot(est, ref) / (np.linalg.norm(est) * np.linalg.norm(ref) + 1e-30)

    # NRMSE: RMS error / dynamic range
    rmse = np.sqrt(np.mean((est - ref) ** 2))
    dynamic_range = ref.max() - ref.min()
    nrmse = rmse / (dynamic_range + 1e-30)

    return {"ncc": float(ncc), "nrmse": float(nrmse)}


def plot_phase_comparison(dpc_phase, parallax_phase, ptycho_phase,
                          save_path=None):
    """Side-by-side comparison of DPC, parallax, and ptychographic phase.

    Parameters
    ----------
    dpc_phase : np.ndarray
    parallax_phase : np.ndarray
    ptycho_phase : np.ndarray
    save_path : str, optional
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    titles = ["DPC", "Parallax", "Ptychography"]
    images = [dpc_phase, parallax_phase, ptycho_phase]

    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img, cmap="magma")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_reconstruction(object_phase, probe_complex, error_history,
                        save_path=None):
    """Plot ptychographic reconstruction summary.

    Parameters
    ----------
    object_phase : np.ndarray
    probe_complex : np.ndarray
    error_history : list of float
    save_path : str, optional
    """
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(object_phase, cmap="magma")
    ax1.set_title("Reconstructed object phase")
    ax1.set_xlabel("y [pixels]")
    ax1.set_ylabel("x [pixels]")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 2, 2)
    probe_intensity = np.abs(probe_complex) ** 2
    ax2.imshow(probe_intensity, cmap="gray")
    ax2.set_title("Probe intensity")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(error_history)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("NMSE")
    ax3.set_title("Convergence")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_virtual_images(bf, df, save_path=None):
    """Plot bright-field and dark-field virtual images.

    Parameters
    ----------
    bf : np.ndarray, shape (Rx, Ry)
    df : np.ndarray, shape (Rx, Ry)
    save_path : str, optional
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(bf, cmap="viridis")
    axes[0].set_title("Bright Field")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(df, cmap="viridis")
    axes[1].set_title("Dark Field")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def print_metrics_table(metrics_dict):
    """Print a formatted metrics comparison table.

    Parameters
    ----------
    metrics_dict : dict
        Keys are method names, values are dicts with 'ncc' and 'nrmse'.
    """
    print(f"{'Method':<30s} {'NCC':>8s} {'NRMSE':>8s}")
    print("-" * 50)
    for method, m in metrics_dict.items():
        print(f"{method:<30s} {m['ncc']:8.4f} {m['nrmse']:8.4f}")
