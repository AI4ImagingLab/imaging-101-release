"""Visualisation utilities for SMLFM results.

All functions accept a matplotlib Figure and return it.
Never call matplotlib.use() here — backend selection belongs in main.py.
"""

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from .physics_model import FourierMicroscope, Localisations, MicroLensArray


def plot_mla_alignment(
    fig: Figure,
    lfl: Localisations,
    mla: MicroLensArray,
    lfm: FourierMicroscope,
) -> Figure:
    """Overlay 2D localisations with microlens centres for alignment check.

    Each localisation is coloured by its assigned lens index so you can
    see immediately whether the MLA rotation/offset is correct.

    Args:
        fig: Matplotlib Figure to draw into.
        lfl: Localisations object (after assign_to_lenses).
        mla: Rotated MicroLensArray.
        lfm: FourierMicroscope.

    Returns:
        The same Figure with the alignment plot.
    """
    ax = fig.add_subplot(111)
    lens_centres_xy = (mla.lens_centres - mla.centre) * lfm.mla_to_xy_scale

    sample = lfl.locs_2d[::10]
    ax.scatter(sample[:, 3], sample[:, 4], s=0.3,
               c=sample[:, 12], cmap="tab20", alpha=0.4)
    ax.scatter(lens_centres_xy[:, 0], lens_centres_xy[:, 1],
               s=150, c="k", marker="+", linewidths=1.5, label="Lens centres")
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title("MLA alignment: localisations coloured by assigned lens")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_3d_locs(
    fig: Figure,
    locs_3d: npt.NDArray[float],
    max_lateral_err: float = None,
    min_views: int = None,
) -> Figure:
    """3D scatter plot of reconstructed molecule positions.

    Molecules are coloured by Z coordinate; axis aspect is equal.

    Args:
        fig:             Matplotlib Figure to draw into.
        locs_3d:         (M, 8) array from solvers.fit_3d_localizations.
        max_lateral_err: Exclude molecules with lateral error above this (µm).
        min_views:       Exclude molecules seen in fewer views.

    Returns:
        The same Figure.
    """
    keep = _quality_mask(locs_3d, max_lateral_err, min_views)
    xyz  = locs_3d[keep, 0:3]

    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    s=1, c=xyz[:, 2], cmap="RdYlBu_r", marker="o")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Z (µm)")
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (µm)")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_histograms(
    fig: Figure,
    locs_3d: npt.NDArray[float],
    max_lateral_err: float = None,
    min_views: int = None,
) -> Figure:
    """Three-panel histogram: lateral error, axial error, photon count.

    Args:
        fig:             Matplotlib Figure.
        locs_3d:         (M, 8) array from fit_3d_localizations.
        max_lateral_err: Optional lateral error filter threshold (µm).
        min_views:       Optional minimum view count filter.

    Returns:
        The same Figure.
    """
    keep     = _quality_mask(locs_3d, max_lateral_err, min_views)
    lat_nm   = locs_3d[keep, 3] * 1000
    ax_nm    = locs_3d[keep, 4] * 1000
    photons  = locs_3d[keep, 6]

    axes = fig.subplots(1, 3)

    axes[0].hist(lat_nm, bins=np.arange(0, 210, 5), color="steelblue", edgecolor="none")
    axes[0].axvline(np.median(lat_nm), color="r", ls="--",
                    label=f"Median = {np.median(lat_nm):.0f} nm")
    axes[0].set_xlabel("Lateral fit error (nm)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Lateral precision")
    axes[0].legend()

    axes[1].hist(ax_nm, bins=np.arange(0, 300, 5), color="coral", edgecolor="none")
    axes[1].axvline(np.median(ax_nm), color="k", ls="--",
                    label=f"Median = {np.median(ax_nm):.0f} nm")
    axes[1].set_xlabel("Axial fit error (nm)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Axial precision")
    axes[1].legend()

    axes[2].hist(photons, bins=50, color="mediumseagreen", edgecolor="none")
    axes[2].axvline(np.median(photons), color="k", ls="--",
                    label=f"Median = {np.median(photons):.0f} ph")
    axes[2].set_xlabel("Photon count")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Signal strength")
    axes[2].legend()

    fig.tight_layout()
    return fig


def plot_occurrences(
    fig: Figure,
    locs_3d: npt.NDArray[float],
    max_lateral_err: float = None,
    min_views: int = None,
) -> Figure:
    """Scatter of lateral vs axial fit error, coloured by photon count.

    Args:
        fig:             Matplotlib Figure.
        locs_3d:         (M, 8) array.
        max_lateral_err: Optional lateral error filter (µm).
        min_views:       Optional minimum view count filter.

    Returns:
        The same Figure.
    """
    keep    = _quality_mask(locs_3d, max_lateral_err, min_views)
    lat_nm  = locs_3d[keep, 3] * 1000
    ax_nm   = locs_3d[keep, 4] * 1000
    photons = locs_3d[keep, 6]

    ax = fig.add_subplot(111)
    sc = ax.scatter(lat_nm, ax_nm, s=1, c=photons, cmap="viridis", alpha=0.5)
    fig.colorbar(sc, ax=ax, label="Photons")
    ax.set_xlabel("Lateral fit error (nm)")
    ax.set_ylabel("Axial fit error (nm)")
    ax.set_title("Precision vs signal strength")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _quality_mask(
    locs_3d: npt.NDArray[float],
    max_lateral_err: float,
    min_views: int,
) -> npt.NDArray[bool]:
    lat_err    = locs_3d[:, 3]
    view_count = locs_3d[:, 5]
    mask = np.ones(locs_3d.shape[0], dtype=bool)
    if max_lateral_err is not None:
        mask &= lat_err < max_lateral_err
    if min_views is not None:
        mask &= view_count > min_views
    return mask
