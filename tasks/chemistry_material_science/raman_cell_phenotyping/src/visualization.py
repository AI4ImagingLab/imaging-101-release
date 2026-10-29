"""
Visualisation utilities and quality metrics for Raman spectral unmixing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ncc(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Normalised cross-correlation (cosine similarity).

    NCC = (est . ref) / (||est|| ||ref||)
    """
    e = estimate.ravel().astype(np.float64)
    r = reference.ravel().astype(np.float64)
    denom = np.linalg.norm(e) * np.linalg.norm(r)
    if denom == 0:
        return 0.0
    return float(np.dot(e, r) / denom)


def compute_nrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    """NRMSE = RMS(est - ref) / (max(ref) - min(ref))."""
    e = estimate.ravel().astype(np.float64)
    r = reference.ravel().astype(np.float64)
    rng = r.max() - r.min()
    if rng == 0:
        return float("inf")
    return float(np.sqrt(np.mean((e - r) ** 2)) / rng)


def compute_metrics(estimate: np.ndarray, reference: np.ndarray) -> dict:
    """Compute standard evaluation metrics."""
    return {
        "ncc": compute_ncc(estimate, reference),
        "nrmse": compute_nrmse(estimate, reference),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_spectra(spectra, spectral_axis, labels=None, title="Raman spectra",
                 stacked=False, ax=None):
    """Plot one or more spectra.

    Parameters
    ----------
    spectra       : ndarray, shape (B,) or (K, B) or list of (B,)
    spectral_axis : ndarray, shape (B,)
    labels        : list of str, optional
    title         : str
    stacked       : bool, if True offset each spectrum vertically
    ax            : matplotlib Axes, optional

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    if isinstance(spectra, np.ndarray) and spectra.ndim == 1:
        spectra = [spectra]
    elif isinstance(spectra, np.ndarray) and spectra.ndim == 2:
        spectra = [spectra[i] for i in range(spectra.shape[0])]

    for i, sp in enumerate(spectra):
        offset = i * 1.0 if stacked else 0
        label = labels[i] if labels else None
        ax.plot(spectral_axis, sp + offset, label=label)

    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(title)
    if labels:
        ax.legend()
    return ax


def plot_band_image(volume_band: np.ndarray, title="", ax=None, cbar=True):
    """Plot a single 2-D band image.

    Parameters
    ----------
    volume_band : ndarray, shape (X, Y)
    """
    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(volume_band, cmap="viridis")
    ax.set_title(title)
    if cbar:
        plt.colorbar(im, ax=ax, shrink=0.7)
    return ax


def plot_abundance_maps(abundance_maps, labels, image_layer=None,
                        cbar=False):
    """Plot abundance maps side by side.

    Parameters
    ----------
    abundance_maps : list of ndarray, each (X, Y, Z) or (X, Y)
    labels         : list of str
    image_layer    : int or None.  If not None, select this z-layer.
    """
    n = len(abundance_maps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, amap, label in zip(axes, abundance_maps, labels):
        img = amap[..., image_layer] if image_layer is not None else amap
        ax.imshow(img, cmap="viridis")
        ax.set_title(label)
        if cbar:
            plt.colorbar(ax.images[0], ax=ax, shrink=0.7)
    plt.tight_layout()
    return fig, axes


def plot_merged_reconstruction(abundance_maps, labels, image_layer,
                               ax=None):
    """Overlay abundance maps in distinct colours for a single z-layer.

    Parameters
    ----------
    abundance_maps : list of ndarray, each (X, Y, Z)
    labels         : list of str
    image_layer    : int
    """
    if ax is None:
        fig, ax = plt.subplots()

    cmap_vals = plt.colormaps["tab10"](np.linspace(0, 1, len(abundance_maps)))
    white = [1, 1, 1, 0]

    order = ["Background", "Cytoplasm", "Nucleus", "Lipids"]
    for lbl in order:
        if lbl not in labels:
            continue
        i = labels.index(lbl)
        cmap_i = LinearSegmentedColormap.from_list("", [white, cmap_vals[i]])
        ax.imshow(abundance_maps[i][..., image_layer], cmap=cmap_i)

    ax.set_title("Merged")
    return ax
