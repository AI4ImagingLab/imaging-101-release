"""
Visualization: metrics computation and plotting utilities.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(estimate, reference):
    """Compute NCC and NRMSE between estimate and reference volumes.

    For complex-valued volumes, metrics are computed on magnitudes.

    Parameters
    ----------
    estimate : numpy.ndarray
        Estimated volume (complex or real).
    reference : numpy.ndarray
        Reference/ground-truth volume (complex or real).

    Returns
    -------
    dict
        'ncc': normalized cross-correlation (cosine similarity).
        'nrmse': normalized root mean squared error (RMS / dynamic range).
    """
    est_mag = np.abs(estimate).ravel().astype(np.float64)
    ref_mag = np.abs(reference).ravel().astype(np.float64)

    # NCC: cosine similarity (no mean subtraction)
    dot = np.dot(est_mag, ref_mag)
    norm_est = np.linalg.norm(est_mag)
    norm_ref = np.linalg.norm(ref_mag)
    if norm_est == 0 or norm_ref == 0:
        ncc = 0.0
    else:
        ncc = float(dot / (norm_est * norm_ref))

    # NRMSE: RMS error / dynamic range of reference
    rms = np.sqrt(np.mean((est_mag - ref_mag) ** 2))
    dynamic_range = float(ref_mag.max() - ref_mag.min())
    if dynamic_range == 0:
        nrmse = float('inf')
    else:
        nrmse = float(rms / dynamic_range)

    return {
        'ncc': ncc,
        'nrmse': nrmse,
    }


def plot_complex_slice(volume_slice, title=None):
    """Plot real and imaginary parts of a 2D complex slice side by side.

    Parameters
    ----------
    volume_slice : numpy.ndarray, 2D complex
        A single 2D slice of a complex volume.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    im1 = ax1.imshow(volume_slice.real)
    ax1.set_title("Real")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(volume_slice.imag)
    ax2.set_title("Imaginary")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_volume_slices(volume, slice_indices):
    """Plot real and imaginary parts for multiple axial slices.

    Parameters
    ----------
    volume : numpy.ndarray, (nz, n, n) complex
        3D complex volume.
    slice_indices : list of int
        Axial (z-axis) indices to plot.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with rows for each slice, columns for real/imaginary.
    """
    n_slices = len(slice_indices)
    fig, axes = plt.subplots(n_slices, 2, figsize=(10, 4 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(slice_indices):
        s = volume[idx]
        im_real = axes[i, 0].imshow(s.real)
        axes[i, 0].set_title(f'Real (z={idx})')
        plt.colorbar(im_real, ax=axes[i, 0], orientation='horizontal')

        im_imag = axes[i, 1].imshow(s.imag)
        axes[i, 1].set_title(f'Imag (z={idx})')
        plt.colorbar(im_imag, ax=axes[i, 1], orientation='horizontal')

    fig.tight_layout()
    return fig
