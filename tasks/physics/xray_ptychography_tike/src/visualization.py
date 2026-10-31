"""Visualization and metric computation for ptychography results."""

import numpy as np


def compute_ncc(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Compute normalized cross-correlation (cosine similarity).

    NCC = (estimate . reference) / (||estimate|| * ||reference||)

    Parameters
    ----------
    estimate : np.ndarray
        Estimated array (flattened internally).
    reference : np.ndarray
        Reference array (flattened internally).

    Returns
    -------
    float
        NCC value in [-1, 1].
    """
    e = estimate.ravel().astype(np.float64)
    r = reference.ravel().astype(np.float64)
    norm_e = np.linalg.norm(e)
    norm_r = np.linalg.norm(r)
    if norm_e == 0 or norm_r == 0:
        return 0.0
    return float(np.dot(e, r) / (norm_e * norm_r))


def compute_nrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Compute normalized root-mean-square error.

    NRMSE = RMSE / (max(reference) - min(reference))

    Parameters
    ----------
    estimate : np.ndarray
        Estimated array.
    reference : np.ndarray
        Reference array.

    Returns
    -------
    float
        NRMSE value (non-negative).
    """
    e = estimate.ravel().astype(np.float64)
    r = reference.ravel().astype(np.float64)
    rmse = np.sqrt(np.mean((e - r) ** 2))
    dynamic_range = r.max() - r.min()
    if dynamic_range == 0:
        return float("inf")
    return float(rmse / dynamic_range)


def compute_metrics(estimate_phase: np.ndarray,
                    reference_phase: np.ndarray) -> dict:
    """Compute evaluation metrics between estimated and reference phase.

    Parameters
    ----------
    estimate_phase : np.ndarray
        Phase of the reconstructed object.
    reference_phase : np.ndarray
        Phase from the baseline reference.

    Returns
    -------
    dict
        Dictionary with 'ncc' and 'nrmse' keys.
    """
    return {
        "ncc": compute_ncc(estimate_phase, reference_phase),
        "nrmse": compute_nrmse(estimate_phase, reference_phase),
    }


def plot_phase(psi_2d, title="Reconstructed Object Phase", save_path=None):
    """Plot amplitude and phase of a 2D complex array side by side.

    Parameters
    ----------
    psi_2d : (W, H) complex64
        A single 2D slice of the reconstructed object.
    title : str
        Plot title.
    save_path : str or None
        If provided, save the figure to this path.
    """
    import matplotlib.pyplot as plt

    amplitude = np.abs(psi_2d)
    phase = np.angle(psi_2d)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    im1 = ax1.imshow(amplitude, cmap='viridis')
    ax1.set_title("Amplitude")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax2.set_title("Phase")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost_curve(costs, title="Reconstruction Cost", save_path=None):
    """Plot the convergence curve of the reconstruction.

    Parameters
    ----------
    costs : array-like
        Per-iteration cost values. Can be 1D or 2D (iterations x batches).
    title : str
        Plot title.
    save_path : str or None
        If provided, save the figure to this path.
    """
    import matplotlib.pyplot as plt

    costs = np.array(costs)
    if costs.ndim == 2:
        # Average across batches for plotting
        cost_per_iter = costs.mean(axis=1)
    else:
        cost_per_iter = costs

    plt.figure(dpi=150)
    plt.semilogy(cost_per_iter)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
