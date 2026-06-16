"""Visualization utilities and SNR computation for ADI exoplanet imaging.

Provides:
- `plot_raw_frame`: Plot a single ADI frame with IWA circle and scalebar.
- `plot_klip_result`: Plot the KLIP final image with companion annotation.
- `compute_snr`: Compute the signal-to-noise ratio of a detected companion
  using the Mawet et al. (2014) two-sample t-test formulation.
"""

from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_raw_frame(
    frame: np.ndarray,
    center=None,
    iwa: float = None,
    vmin: float = None,
    vmax: float = None,
    log_scale: bool = True,
    scalebar_length: float = None,
    scalebar_label: str = None,
    title: str = None,
    output_path: str = None,
    figsize: tuple = (5, 5),
):
    """Plot a single ADI frame with optional IWA circle and scalebar.

    Parameters
    ----------
    frame : np.ndarray, shape (H, W)
    center : (cx, cy), optional.  Defaults to image centre.
    iwa : float, optional.  IWA radius in pixels for dashed circle.
    vmin, vmax : float, optional.  Colormap range.
    log_scale : bool.  Use logarithmic normalization.
    scalebar_length : float, optional.  Scalebar length in pixels.
    scalebar_label : str, optional.  Scalebar annotation text.
    title : str, optional.
    output_path : str, optional.  Save figure to this path.
    figsize : tuple.

    Returns
    -------
    fig, ax : matplotlib objects.
    """
    if center is None:
        center = ((frame.shape[1] - 1) / 2.0, (frame.shape[0] - 1) / 2.0)

    fig, ax = plt.subplots(1, figsize=figsize)
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
    kwargs = dict(origin='lower', cmap='inferno')
    if norm is not None:
        im = ax.imshow(frame, norm=norm, **kwargs)
    else:
        im = ax.imshow(frame, vmin=vmin, vmax=vmax, **kwargs)

    if iwa is not None:
        circle = Circle(center, iwa, facecolor='none',
                        edgecolor='white', lw=1.5, linestyle='--', alpha=0.7)
        ax.add_patch(circle)

    if scalebar_length is not None and scalebar_label is not None:
        _add_scalebar(ax, scalebar_length, scalebar_label)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(im, cax=cax, label='Counts')
    ax.set_xlabel('$x$ pixel')
    ax.set_ylabel('$y$ pixel')
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_klip_result(
    image: np.ndarray,
    center=None,
    iwa: float = None,
    vmin: float = None,
    vmax: float = None,
    scalebar_length: float = None,
    scalebar_label: str = None,
    planet_xy=None,
    title: str = None,
    output_path: str = None,
    figsize: tuple = (5, 5),
    xlim_half: float = None,
    ylim_half: float = None,
):
    """Plot a KLIP-processed image with companion marker and IWA circle.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
    center : (cx, cy), optional.
    iwa : float, optional.
    vmin, vmax : float, optional.  Linear colormap range.
    scalebar_length, scalebar_label : optional scalebar.
    planet_xy : (px, py), optional.  Mark companion position with a cross.
    title : str, optional.
    output_path : str, optional.
    figsize : tuple.
    xlim_half, ylim_half : float, optional.  Zoom half-range in pixels.

    Returns
    -------
    fig, ax : matplotlib objects.
    """
    if center is None:
        center = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)

    fig, ax = plt.subplots(1, figsize=figsize)
    im = ax.imshow(image, origin='lower', vmin=vmin, vmax=vmax, cmap='RdBu_r')

    if iwa is not None:
        circle = Circle(center, iwa, facecolor='none',
                        edgecolor='white', lw=1.5, linestyle='--', alpha=0.7)
        ax.add_patch(circle)

    if planet_xy is not None:
        ax.plot(planet_xy[0], planet_xy[1], '+', color='cyan',
                markersize=12, markeredgewidth=1.5, label='Beta Pic b')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.6)

    if scalebar_length is not None and scalebar_label is not None:
        _add_scalebar(ax, scalebar_length, scalebar_label)

    if xlim_half is not None:
        ax.set_xlim(center[0] - xlim_half, center[0] + xlim_half)
    if ylim_half is not None:
        ax.set_ylim(center[1] - ylim_half, center[1] + ylim_half)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(im, cax=cax, label='Counts')
    ax.set_xlabel('$x$ pixel')
    ax.set_ylabel('$y$ pixel')
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig, ax


def _add_scalebar(ax, length_px: float, label: str, color: str = 'white'):
    """Add a simple scalebar annotation to an existing axis."""
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    scalebar = AnchoredSizeBar(
        ax.transData, length_px, label, 'lower right',
        pad=0.5, sep=5, color=color, frameon=False,
        size_vertical=0.3,
        fontproperties=fm.FontProperties(size=9),
    )
    ax.add_artist(scalebar)


# ---------------------------------------------------------------------------
# SNR computation (Mawet et al. 2014)
# ---------------------------------------------------------------------------

def _get_aperture_centers(
    r_px: float,
    pa_deg: float,
    fwhm: float,
    exclude_nearest: int = 0,
) -> List[Tuple[float, float]]:
    """Place apertures in a ring at radius r_px, spaced by fwhm.

    Returns list of (offset_x, offset_y) from image centre, starting at pa_deg.
    The first aperture is at pa_deg (the companion position); the rest are noise.
    Apertures within `exclude_nearest` steps of the companion are omitted.
    """
    n = max(1, int(2 * np.pi * r_px / fwhm))
    start = np.deg2rad(pa_deg + 90)
    delta = 2 * np.pi / n
    centers = []
    # First: companion aperture
    centers.append((r_px * np.cos(start), r_px * np.sin(start)))
    # Remaining: noise apertures, skipping exclude_nearest on each side
    for i in range(1 + exclude_nearest, n - exclude_nearest):
        theta = start + i * delta
        centers.append((r_px * np.cos(theta), r_px * np.sin(theta)))
    return centers


def _aperture_value(
    image: np.ndarray,
    cx_img: float,
    cy_img: float,
    offset_x: float,
    offset_y: float,
    radius: float,
) -> float:
    """Compute the median within a circular aperture centred at (cx+ox, cy+oy)."""
    H, W = image.shape
    y, x = np.mgrid[:H, :W]
    dx = x - (cx_img + offset_x)
    dy = y - (cy_img + offset_y)
    mask = (dx ** 2 + dy ** 2) <= radius ** 2
    vals = image[mask]
    return float(np.nanmedian(vals))


def compute_snr(
    image: np.ndarray,
    planet_x: float,
    planet_y: float,
    fwhm: float,
    exclude_nearest: int = 0,
) -> float:
    """Compute companion SNR using the Mawet et al. (2014) t-test formulation.

    Apertures of diameter `fwhm` are placed in a ring at the companion's
    separation.  The first aperture is centred on the companion; the rest
    sample the background noise.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
    planet_x, planet_y : float
        Companion pixel coordinates (x = column, y = row).
    fwhm : float
        PSF FWHM in pixels (used as aperture diameter).
    exclude_nearest : int
        Number of noise apertures adjacent to the companion to skip.

    Returns
    -------
    snr : float
    """
    H, W = image.shape
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    dx = planet_x - cx
    dy = planet_y - cy
    r_px = np.hypot(dx, dy)
    pa_deg = float(np.degrees(np.arctan2(dy, dx)) - 90.0)
    aperture_r = fwhm / 2.0

    centers = _get_aperture_centers(r_px, pa_deg, fwhm, exclude_nearest)
    values = [
        _aperture_value(image, cx, cy, ox, oy, aperture_r)
        for ox, oy in centers
    ]
    signal = values[0]
    noises = np.array(values[1:], dtype=np.float64)
    n = len(noises)
    if n < 2:
        return float('nan')
    snr = (signal - noises.mean()) / (noises.std(ddof=1) * np.sqrt(1 + 1.0 / n))
    return float(snr)
