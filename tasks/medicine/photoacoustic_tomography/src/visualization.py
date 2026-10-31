"""
Visualization utilities and metrics for photoacoustic tomography.
"""

import numpy as np


def compute_ncc(estimate, reference):
    """Compute normalised cross-correlation (cosine similarity).

    Parameters
    ----------
    estimate : np.ndarray
        Reconstructed image.
    reference : np.ndarray
        Reference image.

    Returns
    -------
    float
        NCC value in [0, 1] for non-negative images.
    """
    e = estimate.ravel().astype(np.float64)
    r = reference.ravel().astype(np.float64)
    denom = np.linalg.norm(e) * np.linalg.norm(r)
    if denom == 0:
        return 0.0
    return float(np.dot(e, r) / denom)


def compute_nrmse(estimate, reference):
    """Compute normalised root-mean-square error.

    NRMSE = RMS(estimate - reference) / (max(reference) - min(reference))

    Parameters
    ----------
    estimate : np.ndarray
        Reconstructed image.
    reference : np.ndarray
        Reference image.

    Returns
    -------
    float
        NRMSE value.
    """
    e = estimate.ravel().astype(np.float64)
    r = reference.ravel().astype(np.float64)
    dynamic_range = r.max() - r.min()
    if dynamic_range == 0:
        return float('inf')
    rmse = np.sqrt(np.mean((e - r)**2))
    return float(rmse / dynamic_range)


def centre_crop(image, fraction=0.8):
    """Extract central crop of a 2D image.

    Parameters
    ----------
    image : np.ndarray, shape (nx, ny) or (nx, ny, 1)
        Input image.
    fraction : float
        Fraction of each dimension to keep.

    Returns
    -------
    np.ndarray
        Cropped image (always 2D).
    """
    img = np.squeeze(image)
    nx, ny = img.shape
    cx = int(nx * (1 - fraction) / 2)
    cy = int(ny * (1 - fraction) / 2)
    return img[cx:nx - cx, cy:ny - cy]


def plot_reconstruction(recon, xf, yf, ax=None, title="Reconstruction",
                        dynamic_range_db=6):
    """Plot 2D reconstruction image in dB scale.

    Parameters
    ----------
    recon : np.ndarray, shape (nx, ny) or (nx, ny, 1)
        Normalised reconstruction.
    xf : np.ndarray
        X-coordinates in meters.
    yf : np.ndarray
        Y-coordinates in meters.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates new figure.
    title : str
        Plot title.
    dynamic_range_db : float
        Dynamic range in dB for display.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    img = np.squeeze(recon)
    img_db = 20 * np.log10(np.clip(np.abs(img), 1e-10, None))
    img_db = np.transpose(img_db)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    extent = [xf[0] * 1e3, xf[-1] * 1e3, yf[-1] * 1e3, yf[0] * 1e3]
    im = ax.imshow(img_db, extent=extent, vmin=-dynamic_range_db, vmax=0,
                   cmap='gray')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='dB')
    return ax


def plot_cross_sections(recon, xf, yf, axes=None):
    """Plot orthogonal cross-sections through the image centre.

    Parameters
    ----------
    recon : np.ndarray, shape (nx, ny) or (nx, ny, 1)
        Normalised reconstruction.
    xf : np.ndarray
        X-coordinates in meters.
    yf : np.ndarray
        Y-coordinates in meters.
    axes : tuple of two Axes or None

    Returns
    -------
    axes : tuple of Axes
    """
    import matplotlib.pyplot as plt

    img = np.squeeze(recon)

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax1, ax2 = axes

    x_mid = len(xf) // 2
    y_mid = len(yf) // 2

    ax1.plot(yf * 1e3, img[x_mid, :])
    ax1.set_xlabel('y (mm)')
    ax1.set_title(f'x = {xf[x_mid]*1e3:.1f} mm')
    ax1.set_ylabel('Normalised amplitude')

    ax2.plot(xf * 1e3, img[:, y_mid])
    ax2.set_xlabel('x (mm)')
    ax2.set_title(f'y = {yf[y_mid]*1e3:.1f} mm')
    ax2.set_ylabel('Normalised amplitude')

    return axes


def plot_signals(signals, t, xd, yd, det_indices=None, ax=None):
    """Plot time-domain PA signals for selected detectors.

    Parameters
    ----------
    signals : np.ndarray, shape (n_time, n_det_x, n_det_y)
        PA time-series.
    t : np.ndarray, shape (n_time,)
        Time vector in seconds.
    xd : np.ndarray
        Detector x-positions.
    yd : np.ndarray
        Detector y-positions.
    det_indices : list of (int, int) or None
        Detector indices to plot. If None, plots centre detector.
    ax : Axes or None

    Returns
    -------
    ax : Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    if det_indices is None:
        cx = len(xd) // 2
        cy = len(yd) // 2
        det_indices = [(cx, cy)]

    for xi, yi in det_indices:
        label = f"det ({xd[xi]*1e3:.1f}, {yd[yi]*1e3:.1f}) mm"
        ax.plot(t * 1e6, signals[:, xi, yi], label=label)

    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Pressure (a.u.)')
    ax.set_title('PA Time-Series')
    ax.legend()
    return ax
