"""Visualization utilities and quantitative metrics for CT reconstruction."""

import numpy as np


def compute_metrics(estimate, reference):
    """Compute reconstruction quality metrics.

    Parameters
    ----------
    estimate : ndarray
        Reconstructed image.
    reference : ndarray
        Reference image (ground truth or baseline reconstruction).

    Returns
    -------
    dict with keys:
        nrmse : float
            Normalised Root Mean Square Error (RMSE / dynamic range).
        ncc : float
            Normalised Cross-Correlation (cosine similarity, no mean subtraction).
    """
    est = estimate.ravel().astype(np.float64)
    ref = reference.ravel().astype(np.float64)

    # NRMSE: RMS error / dynamic range
    rmse = np.sqrt(np.mean((est - ref) ** 2))
    dynamic_range = ref.max() - ref.min()
    nrmse = rmse / dynamic_range if dynamic_range > 0 else float('inf')

    # NCC: cosine similarity (no mean subtraction)
    norm_est = np.linalg.norm(est)
    norm_ref = np.linalg.norm(ref)
    if norm_est == 0 or norm_ref == 0:
        ncc = 0.0
    else:
        ncc = float(np.dot(est, ref) / (norm_est * norm_ref))

    return {"nrmse": nrmse, "ncc": ncc}


def plot_sinogram(sinogram, ax=None, title="Sinogram"):
    """Plot a sinogram.

    Parameters
    ----------
    sinogram : ndarray, shape (n_angles, n_detector)
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(sinogram, aspect='auto', cmap='gray')
    ax.set_xlabel("Detector pixel")
    ax.set_ylabel("Projection index")
    ax.set_title(title)
    return ax


def plot_reconstruction(image, ax=None, title="Reconstruction"):
    """Plot a reconstructed image.

    Parameters
    ----------
    image : ndarray, shape (n, n)
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_comparison(images, titles, suptitle="Comparison"):
    """Plot multiple images side-by-side.

    Parameters
    ----------
    images : list of ndarray
    titles : list of str
    suptitle : str

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig
