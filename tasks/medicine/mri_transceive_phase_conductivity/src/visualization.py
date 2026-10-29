"""Shared review metrics and mask-aware central slices; no backend mutation."""
import numpy as np
import matplotlib.pyplot as plt


def agreement(estimate, reference, mask):
    """Cosine NCC and reference-dynamic-range NRMSE on exactly given mask."""
    x = np.asarray(estimate, dtype=float)[mask].ravel()
    y = np.asarray(reference, dtype=float)[mask].ravel()
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('Nonfinite metric input')
    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0 or np.ptp(y) == 0:
        raise ValueError('Degenerate metric input')
    return {'ncc': float(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)),
            'nrmse': float(np.sqrt(np.mean((x-y)**2)) / np.ptp(y))}


def slice_image(ax, volume, mask, title, *, cmap='viridis', limits=None, unit='S/m', z=None):
    """Plot actual native axial plane using millimetres in x/y coordinates."""
    if z is None:
        z = int(np.round(np.argwhere(mask)[:, 0].mean()))
    values = np.ma.array(volume[z], mask=~mask[z])
    image = ax.imshow(values, origin='lower', extent=(-80, 80, -96, 96), cmap=cmap,
                      vmin=None if limits is None else limits[0], vmax=None if limits is None else limits[1],
                      rasterized=True)
    ax.set(title=title, xlabel='x (mm)', ylabel='y (mm)')
    plt.colorbar(image, ax=ax, label=unit, shrink=.8)
    return image
