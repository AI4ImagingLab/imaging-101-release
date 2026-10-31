"""visualization.py — B-mode display and quality metrics."""

import numpy as np
from scipy.signal import hilbert


def envelope_bmode(migSIG: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Compute envelope-detected, power-law-compressed B-mode image.

    The Hilbert transform is applied along the time axis (axis=0) to the
    real part of the coherently compounded RF image, yielding the analytic
    signal.  The envelope is then raised to the power gamma for display.

    Parameters
    ----------
    migSIG : np.ndarray, shape (N_t, N_x), complex
        Coherently compounded migrated RF image.
    gamma : float
        Power-law exponent for dynamic range compression (default 0.5).

    Returns
    -------
    np.ndarray, shape (N_t, N_x), float64
        B-mode image (non-negative).
    """
    env = np.abs(hilbert(np.real(migSIG), axis=0))
    return env ** gamma


def plot_bmode(bmode: np.ndarray, x: np.ndarray, z: np.ndarray,
               title: str = '', ax=None):
    """Display B-mode image with correct physical axes (x in m, z in m).

    Parameters
    ----------
    bmode : np.ndarray, shape (N_t, N_x)
        B-mode image from envelope_bmode().
    x : np.ndarray, shape (N_x,)
        Lateral positions (m).
    z : np.ndarray, shape (N_t,)
        Depth positions (m).
    title : str
        Plot title.
    ax : matplotlib.axes.Axes or None
        If None, the current active axes is used.

    Returns
    -------
    matplotlib.image.AxesImage
    """
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    dx = np.mean(np.diff(x))
    dz = np.mean(np.diff(z))
    ext = (x.min() - dx, x.max() + dx,
           z.max() + dz, z.min() - dz)
    im = ax.imshow(bmode, cmap='gray', extent=ext,
                   aspect='equal', interpolation='none')
    ax.set_xlabel('Azimuth (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_title(title)
    return im


def measure_psf_fwhm(bmode: np.ndarray, x: np.ndarray, z: np.ndarray,
                     z_targets) -> list:
    """Measure lateral FWHM (mm) of point-spread function at each target depth.

    For each target depth, the row closest to z_target is extracted.
    The FWHM is estimated from the lateral profile around the peak.

    Parameters
    ----------
    bmode : np.ndarray, shape (N_t, N_x)
    x : np.ndarray, shape (N_x,)
    z : np.ndarray, shape (N_t,)
    z_targets : list of float
        Approximate depths of wire targets (m).

    Returns
    -------
    list of float
        Lateral FWHM in mm for each target.
    """
    fwhms = []
    for zt in z_targets:
        iz = int(np.argmin(np.abs(z - zt)))
        row = bmode[iz, :]
        ipeak = int(np.argmax(row))
        half_max = row[ipeak] / 2.0
        # Find left crossing
        left = ipeak
        while left > 0 and row[left] >= half_max:
            left -= 1
        # Find right crossing
        right = ipeak
        while right < len(row) - 1 and row[right] >= half_max:
            right += 1
        fwhm_m = (x[right] - x[left])
        fwhms.append(float(fwhm_m * 1e3))   # convert to mm
    return fwhms


def measure_cnr(bmode: np.ndarray, x: np.ndarray, z: np.ndarray,
                cyst_centers: list,
                cyst_radius: float = 2e-3,
                shell_inner: float = 2.5e-3,
                shell_outer: float = 4e-3) -> list:
    """Measure contrast-to-noise ratio (CNR) for each cyst.

    CNR = |mean_inside - mean_outside| / std_outside

    where 'inside' is a disk of radius cyst_radius centred on cyst_center
    and 'outside' is an annular shell from shell_inner to shell_outer.

    Parameters
    ----------
    bmode : np.ndarray, shape (N_t, N_x)
    x : np.ndarray, shape (N_x,)
    z : np.ndarray, shape (N_t,)
    cyst_centers : list of (x_c, z_c) tuples (m)
    cyst_radius : float (m)
    shell_inner, shell_outer : float (m)

    Returns
    -------
    list of float
    """
    X, Z = np.meshgrid(x, z)
    cnrs = []
    for (xc, zc) in cyst_centers:
        dist = np.sqrt((X - xc) ** 2 + (Z - zc) ** 2)
        inside  = dist <= cyst_radius
        outside = (dist > shell_inner) & (dist <= shell_outer)
        if inside.sum() == 0 or outside.sum() == 0:
            cnrs.append(float('nan'))
            continue
        mu_in  = bmode[inside].mean()
        mu_out = bmode[outside].mean()
        sg_out = bmode[outside].std()
        cnr    = abs(mu_in - mu_out) / (sg_out + np.spacing(1))
        cnrs.append(float(cnr))
    return cnrs
