"""
Visualization utilities for conventional ptychography (CP).

All functions are backend-agnostic: they accept Axes objects and do NOT
call matplotlib.use() or plt.show(). Backend selection is main.py's
responsibility; notebooks use %matplotlib inline.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes


def complex_to_hsv(arr: np.ndarray, max_amp: float = None) -> np.ndarray:
    """
    Convert a complex array to an HSV-encoded RGB image.

    Encoding:
    - Hue    = phase angle (0 → 2π maps to 0 → 1)
    - Value  = normalized amplitude (0 → max_amp maps to 0 → 1)
    - Saturation = 1 (constant, fully saturated)

    This is the standard false-color representation for complex-valued
    reconstructions in ptychography (quantitative phase imaging).

    Parameters
    ----------
    arr : ndarray, shape (..., Ny, Nx), complex
        Complex array (e.g., reconstructed object or probe). Leading
        dimensions are squeezed.
    max_amp : float, optional
        Maximum amplitude for normalization. Defaults to the array's max.

    Returns
    -------
    rgb : ndarray, shape (Ny, Nx, 3), float
        RGB image in [0, 1].
    """
    arr2d = np.squeeze(arr)
    amp = np.abs(arr2d)
    phase = np.angle(arr2d)  # in [-π, π]

    if max_amp is None:
        max_amp = amp.max() + 1e-12

    hue = (phase + np.pi) / (2 * np.pi)   # normalize to [0, 1]
    sat = np.ones_like(hue)
    val = np.clip(amp / max_amp, 0, 1)

    hsv = np.stack([hue, sat, val], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def plot_complex_image(
    arr: np.ndarray,
    ax: Axes,
    title: str = "",
    pixel_size_um: float = None,
    max_amp: float = None,
):
    """
    Display a complex array as an HSV-encoded image.

    Parameters
    ----------
    arr : ndarray, complex
        Complex array to display.
    ax : Axes
        Matplotlib axes.
    title : str
        Subplot title.
    pixel_size_um : float, optional
        Pixel size in microns. If given, axes are labelled in μm.
    max_amp : float, optional
        Saturation point for amplitude (defaults to max of array).
    """
    rgb = complex_to_hsv(arr, max_amp=max_amp)
    if pixel_size_um is not None:
        ny, nx = rgb.shape[:2]
        extent = [-nx / 2 * pixel_size_um, nx / 2 * pixel_size_um,
                  ny / 2 * pixel_size_um, -ny / 2 * pixel_size_um]
        ax.imshow(rgb, extent=extent)
        ax.set_xlabel("x [μm]")
        ax.set_ylabel("y [μm]")
    else:
        ax.imshow(rgb)
    ax.set_title(title)


def add_colorwheel(ax: Axes, size: int = 64):
    """
    Add a phase colorwheel inset to the given axes.

    The colorwheel illustrates the hue–phase mapping:
    red = 0, yellow = π/2, cyan = π, blue = -π/2.

    Parameters
    ----------
    ax : Axes
        Axes on which to overlay the inset.
    size : int
        Size of the colorwheel in pixels.
    """
    x = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2)
    mask = r <= 1
    phase = np.arctan2(Y, X)
    amp = r * mask

    hue = (phase + np.pi) / (2 * np.pi)
    hsv = np.stack([hue, np.ones_like(hue), amp * mask], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    rgb[~mask] = 1.0  # white outside circle

    inset = ax.inset_axes([0.72, 0.02, 0.25, 0.25])
    inset.imshow(rgb, interpolation="bilinear")
    inset.axis("off")
    inset.set_title("π\n", fontsize=6, pad=0)


def plot_diffraction_pattern(
    pattern: np.ndarray,
    ax: Axes,
    title: str = "diffraction pattern",
    log_scale: bool = True,
):
    """
    Display a single diffraction pattern.

    Parameters
    ----------
    pattern : ndarray, shape (Nd, Nd), float
        Intensity pattern.
    ax : Axes
    title : str
    log_scale : bool
        If True, display log(pattern + 1) for better dynamic range.
    """
    data = np.log(np.abs(pattern) + 1) if log_scale else pattern
    ax.imshow(data, cmap="gray")
    ax.set_title(title)
    ax.axis("off")


def plot_scan_grid(
    positions: np.ndarray,
    Np: int,
    obj_shape: tuple,
    ax: Axes,
    title: str = "scan grid",
):
    """
    Overlay scan positions on the object field of view.

    Parameters
    ----------
    positions : ndarray, shape (J, 2), int
        Scan positions (row, col of probe upper-left corner).
    Np : int
        Probe size in pixels.
    obj_shape : tuple
        (No, No) object array shape.
    ax : Axes
    title : str
    """
    centers = positions + Np // 2   # center of each probe window
    ax.plot(centers[:, 1], centers[:, 0], ".", ms=2, alpha=0.6, color="tab:orange")
    ax.set_xlim(0, obj_shape[1])
    ax.set_ylim(obj_shape[0], 0)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("col [px]")
    ax.set_ylabel("row [px]")


def plot_reconstruction_summary(
    obj: np.ndarray,
    probe: np.ndarray,
    error_history: list,
    ptychogram_sample: np.ndarray = None,
    pixel_size_um: float = None,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """
    Create a four-panel summary figure of the CP reconstruction.

    Panels:
    1. Reconstructed object (complex HSV)
    2. Reconstructed probe (complex HSV)
    3. Convergence curve (error vs. iteration)
    4. Sample diffraction pattern (if provided)

    Parameters
    ----------
    obj : ndarray, complex
        Reconstructed object array.
    probe : ndarray, complex
        Reconstructed probe array.
    error_history : list of float
        Reconstruction error per iteration.
    ptychogram_sample : ndarray, optional
        A single diffraction pattern to display.
    pixel_size_um : float, optional
        Object pixel size in μm for axis labels.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : Figure
    """
    ncols = 4 if ptychogram_sample is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    plot_complex_image(obj, axes[0], title="object (amplitude & phase)",
                       pixel_size_um=pixel_size_um)
    plot_complex_image(probe, axes[1], title="probe (amplitude & phase)",
                       pixel_size_um=pixel_size_um)

    axes[2].semilogy(error_history, color="steelblue", lw=1.5)
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("error")
    axes[2].set_title("convergence")
    axes[2].grid(True, alpha=0.3)

    if ptychogram_sample is not None:
        plot_diffraction_pattern(ptychogram_sample, axes[3])

    fig.tight_layout()
    return fig
