"""
Visualization utilities for Fourier ptychography (FP).

In FP the reconstructed quantity in the object plane is the high-resolution
*object spectrum* Õ(q) = FT{O(r)}. The real-space object O(r) = IFT{Õ} gives
the quantitative phase image (QPI). The pupil P̃(q) is reconstructed in k-space.

All functions are backend-agnostic (no matplotlib.use() calls).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from .utils import ifft2c

# ------------------------------------------------------------------ #
# HSV complex encoding (shared with CP)
# ------------------------------------------------------------------ #

def complex_to_hsv(arr: np.ndarray, max_amp: float = None) -> np.ndarray:
    """
    Convert a complex array to an HSV-encoded RGB image.

    Hue = phase (0 → 2π → color wheel), Value = normalized amplitude.

    Parameters
    ----------
    arr : ndarray, complex
    max_amp : float, optional

    Returns
    -------
    rgb : ndarray, shape (Ny, Nx, 3), float in [0,1]
    """
    arr2d = np.squeeze(arr)
    amp = np.abs(arr2d)
    phase = np.angle(arr2d)
    if max_amp is None:
        max_amp = amp.max() + 1e-12
    hue = (phase + np.pi) / (2 * np.pi)
    val = np.clip(amp / max_amp, 0, 1)
    hsv = np.stack([hue, np.ones_like(hue), val], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def plot_complex_image(
    arr: np.ndarray,
    ax: Axes,
    title: str = "",
    pixel_size_um: float = None,
    max_amp: float = None,
):
    """Display a complex array as an HSV image."""
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


# ------------------------------------------------------------------ #
# FPM-specific: object spectrum and pupil
# ------------------------------------------------------------------ #

def get_object_realspace(reconstruction) -> np.ndarray:
    """
    Convert the reconstructed k-space object to real-space via ifft2c.

    In FPM, state.object stores the high-resolution k-space spectrum Õ(q).
    This function applies an inverse FFT to obtain the real-space image O(r).

    Parameters
    ----------
    reconstruction : PtyState

    Returns
    -------
    obj_real : ndarray, shape (No, No), complex
    """
    obj_kspace = np.squeeze(reconstruction.object)
    obj_real = ifft2c(obj_kspace)
    return obj_real


def get_pupil(reconstruction) -> np.ndarray:
    """
    Extract the reconstructed pupil from a PtyState.

    Parameters
    ----------
    reconstruction : PtyState

    Returns
    -------
    pupil : ndarray, shape (Np, Np), complex
    """
    return np.squeeze(reconstruction.probe)


def plot_raw_data_mean(
    ptychogram: np.ndarray,
    ax: Axes,
    title: str = "mean raw image",
    log_scale: bool = False,
):
    """
    Display the mean of all FPM low-resolution images.

    The mean image shows the average illumination and provides a quick quality
    check: a well-centered LED array will produce a circular bright-field disk
    in the center with dark-field rings at the edges.

    Parameters
    ----------
    ptychogram : ndarray, shape (J, Nd, Nd)
    ax : Axes
    title : str
    log_scale : bool
    """
    mean_img = np.mean(ptychogram, axis=0)
    data = np.log(mean_img + 1) if log_scale else mean_img
    ax.imshow(np.rot90(data, 2), cmap="gray")
    ax.set_title(title)
    ax.axis("off")


def plot_brightfield_image(
    ptychogram: np.ndarray,
    ax: Axes,
    title: str = "bright-field image",
):
    """
    Display the bright-field image (the image with maximum total intensity,
    corresponding to the on-axis LED).

    Parameters
    ----------
    ptychogram : ndarray, shape (J, Nd, Nd)
    ax : Axes
    """
    brightfield_idx = np.argmax(ptychogram.sum(axis=(1, 2)))
    ax.imshow(np.rot90(ptychogram[brightfield_idx], 2), cmap="gray")
    ax.set_title(title)
    ax.axis("off")


def plot_pupil(
    pupil: np.ndarray,
    ax: Axes,
    title: str = "pupil",
    pixel_size_um: float = None,
):
    """
    Display the reconstructed pupil (complex-valued) as an HSV image.

    The pupil (also called exit pupil or coherent transfer function) encodes
    the aberrations of the imaging objective. Amplitude = 1 inside the NA
    circle (modulation transfer), phase = wavefront aberration (Zernike modes).

    Parameters
    ----------
    pupil : ndarray, shape (Np, Np), complex
    ax : Axes
    title : str
    pixel_size_um : float, optional
        k-space pixel size in units of μm⁻¹ (for axis labels).
    """
    plot_complex_image(pupil, ax, title=title, pixel_size_um=pixel_size_um)


def plot_reconstruction_summary(
    reconstruction,
    experimental_data,
    error_history: list,
    figsize: tuple = (15, 5),
) -> plt.Figure:
    """
    Create a five-panel FPM reconstruction summary figure.

    Panels:
    1. Mean raw data (bright-field + dark-field)
    2. Bright-field image (on-axis LED)
    3. Reconstructed real-space object amplitude
    4. Reconstructed real-space object phase
    5. Reconstructed pupil (amplitude)
    6. Convergence curve

    Parameters
    ----------
    reconstruction : PtyLab Reconstruction
    experimental_data : PtyLab ExperimentalData
    error_history : list of float
    figsize : tuple

    Returns
    -------
    fig : Figure
    """
    fig, axes = plt.subplots(1, 6, figsize=figsize)

    ptychogram = experimental_data.ptychogram
    obj = get_object_realspace(reconstruction)
    pupil = get_pupil(reconstruction)

    # Panel 1: mean raw image
    plot_raw_data_mean(ptychogram, axes[0], title="mean raw data")

    # Panel 2: bright-field
    plot_brightfield_image(ptychogram, axes[1], title="bright-field image")

    # Panel 3: object amplitude
    axes[2].imshow(np.abs(obj), cmap="gray")
    axes[2].set_title("object amplitude")
    axes[2].axis("off")

    # Panel 4: object phase
    axes[3].imshow(np.angle(obj), cmap="hsv", vmin=-np.pi, vmax=np.pi)
    axes[3].set_title("object phase")
    axes[3].axis("off")

    # Panel 5: pupil amplitude
    axes[4].imshow(np.abs(pupil), cmap="gray")
    axes[4].set_title("pupil amplitude")
    axes[4].axis("off")

    # Panel 6: convergence
    if error_history:
        axes[5].semilogy(error_history, color="steelblue", lw=1.5)
        axes[5].set_xlabel("iteration")
        axes[5].set_ylabel("error")
        axes[5].grid(True, alpha=0.3)
    axes[5].set_title("convergence")

    fig.tight_layout()
    return fig
