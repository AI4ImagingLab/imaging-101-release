"""
visualization.py — Plotting utilities for NLOS imaging results.

All functions are backend-agnostic (no matplotlib.use() calls); the caller
is responsible for selecting the backend before importing this module.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def max_projection(vol: np.ndarray) -> tuple:
    """
    Compute three maximum-intensity projections of a volume.

    Parameters
    ----------
    vol : ndarray, shape (Nz, Ny, Nx)
        Reconstructed NLOS volume (depth × y × x).

    Returns
    -------
    (front, top, side) : three 2-D arrays
        front : max over depth  → (Ny, Nx)   (front view, looking into scene)
        top   : max over y      → (Nz, Nx)   (top-down view)
        side  : max over x      → (Nz, Ny)   (side view)
    """
    front = vol.max(axis=0)          # (Ny, Nx)
    top   = vol.max(axis=1)          # (Nz, Nx)
    side  = vol.max(axis=2)          # (Nz, Ny)
    return front, top, side


def _norm(img: np.ndarray) -> np.ndarray:
    """Normalise image to [0, 1]."""
    lo, hi = img.min(), img.max()
    if hi > lo:
        return (img - lo) / (hi - lo)
    return np.zeros_like(img)


# ---------------------------------------------------------------------------
# Single-result three-view display
# ---------------------------------------------------------------------------

def plot_nlos_result(
    vol: np.ndarray,
    wall_size: float,
    bin_resolution: float,
    title: str = '',
    c: float = 3e8,
    ax=None,
) -> plt.Figure:
    """
    Display front / top / side maximum-intensity projections of an NLOS volume.

    Parameters
    ----------
    vol : ndarray, shape (Nt, Ny, Nx)
    wall_size : float
    bin_resolution : float
    title : str
    c : float
    ax : optional pre-created axes array of length 3

    Returns
    -------
    fig : matplotlib Figure
    """
    Nt, Ny, Nx = vol.shape
    range_m = Nt * c * bin_resolution
    hw = wall_size / 2.0
    z_max = range_m / 2.0

    front, top, side = max_projection(vol)

    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    else:
        axes = ax
        fig  = axes[0].figure

    if title:
        fig.suptitle(title, fontsize=13)

    # Front view: looking into the scene  (x horizontal, y vertical)
    axes[0].imshow(_norm(front), cmap='gray', origin='upper',
                   extent=[-hw, hw, hw, -hw], aspect='equal')
    axes[0].set_title('Front view')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')

    # Top view: overhead  (x horizontal, z depth vertical)
    axes[1].imshow(_norm(top), cmap='gray', origin='upper',
                   extent=[-hw, hw, 0, z_max], aspect='auto')
    axes[1].set_title('Top view')
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('z (m)')

    # Side view: from the side  (z depth horizontal, y vertical)
    axes[2].imshow(_norm(side.T), cmap='gray', origin='upper',
                   extent=[0, z_max, hw, -hw], aspect='auto')
    axes[2].set_title('Side view')
    axes[2].set_xlabel('z (m)')
    axes[2].set_ylabel('y (m)')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-method comparison
# ---------------------------------------------------------------------------

def plot_comparison(
    vols: dict,
    wall_size: float,
    bin_resolution: float,
    c: float = 3e8,
    view: str = 'front',
    figsize: tuple | None = None,
) -> plt.Figure:
    """
    Side-by-side comparison of multiple reconstructions (front views by default).

    Parameters
    ----------
    vols : dict[str, ndarray]
        Mapping method-name → volume (Nt, Ny, Nx).
    wall_size : float
    bin_resolution : float
    c : float
    view : {'front', 'top', 'side'}
    figsize : tuple or None

    Returns
    -------
    fig : matplotlib Figure
    """
    n    = len(vols)
    hw   = wall_size / 2.0
    Nt   = next(iter(vols.values())).shape[0]
    z_max = Nt * c * bin_resolution / 2.0
    view_idx = {'front': 0, 'top': 1, 'side': 2}[view]

    if figsize is None:
        figsize = (4 * n, 4)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (name, vol) in zip(axes, vols.items()):
        imgs  = max_projection(vol)
        img   = imgs[view_idx]
        if view == 'front':
            ext = [-hw, hw, hw, -hw]
            xl, yl = 'x (m)', 'y (m)'
        elif view == 'top':
            ext = [-hw, hw, 0, z_max]
            xl, yl = 'x (m)', 'z (m)'
        else:
            img = img.T
            ext = [0, z_max, hw, -hw]
            xl, yl = 'z (m)', 'y (m)'
        ax.imshow(_norm(img), cmap='gray', origin='upper',
                  extent=ext, aspect='equal' if view == 'front' else 'auto')
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Transient measurement display
# ---------------------------------------------------------------------------

def plot_transient(
    meas: np.ndarray,
    wy: int,
    wx: int,
    bin_resolution: float,
    c: float = 3e8,
    title: str = '',
) -> plt.Figure:
    """
    Plot the time-resolved histogram at a single scan point.

    Parameters
    ----------
    meas : ndarray, shape (Nt, Ny, Nx)  (internal format)
    wy, wx : int   — scan-point indices (y, x)
    bin_resolution : float
    c : float
    title : str

    Returns
    -------
    fig : matplotlib Figure
    """
    histogram = meas[:, wy, wx]
    t_ns = np.arange(len(histogram)) * bin_resolution * 1e9   # convert to ns

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t_ns, histogram, lw=0.8)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Photons')
    ax.set_title(title or f'Transient at scan point ({wy}, {wx})')
    ax.set_yscale('log')
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Measurement volume overview  (x-t slice, like paper Fig. 8b)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3-D MIP box  (paper-style cube with three projection faces)
# ---------------------------------------------------------------------------

def plot_volume_3d(
    vol: np.ndarray,
    wall_size: float,
    bin_resolution: float,
    title: str = '',
    c: float = 3e8,
    elev: float = 25,
    azim: float = -50,
    cmap: str = 'hot',
    threshold: float = 0.12,
    gamma_color: float = 0.5,
    gamma_alpha: float = 2.0,
    point_size: float = 1.5,
) -> plt.Figure:
    """
    Volumetric 3-D scatter visualization of a reconstructed NLOS volume.

    Renders the interior of the volume by plotting all voxels above a
    threshold as scatter points, with color and transparency proportional
    to the local intensity.  This shows the full 3-D spatial distribution
    of the reconstructed scene — near and far features are all visible.

    Parameters
    ----------
    vol : ndarray, shape (Nt, Ny, Nx)
        Reconstructed NLOS volume.
    wall_size : float
        Width/height of the relay wall in metres.
    bin_resolution : float
        Temporal bin width in seconds.
    title : str
        Figure title.
    c : float
        Speed of light (m/s).
    elev, azim : float
        3-D view elevation and azimuth angles (degrees).
    cmap : str
        Matplotlib colormap applied to normalised intensity.
    threshold : float
        Fraction of peak intensity below which voxels are not rendered.
    gamma_color : float
        Gamma correction for colour mapping (< 1 brightens mid-tones).
    gamma_alpha : float
        Gamma applied to alpha channel (> 1 makes low-intensity voxels
        more transparent, reducing clutter).
    point_size : float
        Scatter marker size (points²).

    Returns
    -------
    fig : matplotlib Figure
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    Nt, Ny, Nx = vol.shape
    hw    = wall_size / 2.0
    z_max = Nt * c * bin_resolution / 2.0

    # Normalise volume
    vol_max = vol.max()
    if vol_max <= 0:
        vol_n = np.zeros_like(vol)
    else:
        vol_n = vol / vol_max

    # Extract voxels above threshold
    zz, yy, xx = np.where(vol_n > threshold)
    vals = vol_n[zz, yy, xx]

    # Convert voxel indices to physical coordinates
    x_vals = np.linspace(-hw, hw,    Nx)[xx]
    y_vals = np.linspace(-hw, hw,    Ny)[yy]
    z_vals = np.linspace(0,   z_max, Nt)[zz]

    # Colour: apply gamma then map through colormap
    cm   = plt.get_cmap(cmap)
    rgba = cm(vals ** gamma_color).copy()          # (N, 4)
    rgba[:, 3] = np.clip(vals ** gamma_alpha, 0, 1)   # alpha from intensity

    # ── Plot ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(6, 6), facecolor='black')
    ax  = fig.add_subplot(111, projection='3d', facecolor='black')

    # Coordinate mapping: matplotlib (x, y, z) ← physical (wall_x, depth, -wall_y)
    # matplotlib y-axis = depth into scene  (recedes into screen)
    # matplotlib z-axis = -wall_y  so that y=-hw (top of wall) → z=+hw (top of plot)
    #   matching the 2D front-view convention where y=-hw appears at the image top
    ax.scatter(x_vals, z_vals, -y_vals,
               c=rgba, s=point_size,
               linewidths=0, depthshade=False)

    # Bounding-box wireframe — edges in (wall_x, depth, -wall_y) order
    edges = [
        ([-hw,  hw], [0,     0    ], [-hw, -hw]),  # front bottom
        ([-hw,  hw], [0,     0    ], [ hw,  hw]),  # front top
        ([-hw, -hw], [0,     0    ], [-hw,  hw]),  # front left
        ([ hw,  hw], [0,     0    ], [-hw,  hw]),  # front right
        ([-hw,  hw], [z_max, z_max], [-hw, -hw]),  # back bottom
        ([-hw,  hw], [z_max, z_max], [ hw,  hw]),  # back top
        ([-hw, -hw], [z_max, z_max], [-hw,  hw]),  # back left
        ([ hw,  hw], [z_max, z_max], [-hw,  hw]),  # back right
        ([-hw, -hw], [0,     z_max], [-hw, -hw]),  # side edge
        ([ hw,  hw], [0,     z_max], [-hw, -hw]),  # side edge
        ([-hw, -hw], [0,     z_max], [ hw,  hw]),  # side edge
        ([ hw,  hw], [0,     z_max], [ hw,  hw]),  # side edge
    ]
    for xe, ye, ze in edges:
        ax.plot(xe, ye, ze, color=(0.4, 0.4, 0.4), lw=0.5, alpha=0.6)

    # ── Style ───────────────────────────────────────────────────────────────
    ax.set_xlim(-hw,   hw)
    ax.set_ylim(0,     z_max)
    ax.set_zlim(-hw,   hw)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('none')

    ax.tick_params(colors='white', labelsize=7)
    for lbl in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
        lbl.set_color('white')
        lbl.set_fontsize(9)

    ax.set_xlabel('x (m)',    labelpad=4)
    ax.set_ylabel('z (m)',    labelpad=4)   # depth
    ax.set_zlabel('y (m)',    labelpad=4)   # wall vertical
    ax.view_init(elev=elev, azim=azim)

    if title:
        ax.set_title(title, color='white', pad=8, fontsize=11)

    plt.tight_layout()
    return fig


def plot_measurement_slice(
    meas: np.ndarray,
    bin_resolution: float,
    wall_size: float,
    c: float = 3e8,
) -> plt.Figure:
    """
    Show a 2-D slice of the measurement volume: x-axis (wall) vs time.
    Sums over the y dimension to produce a single slice image.
    """
    # meas: (Nt, Ny, Nx) → collapse y
    xt_slice = meas.sum(axis=1)   # (Nt, Nx)
    Nt, Nx   = xt_slice.shape
    hw       = wall_size / 2.0
    t_max_ns = Nt * bin_resolution * 1e9

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(_norm(xt_slice), cmap='hot', origin='upper',
              extent=[-hw, hw, t_max_ns, 0], aspect='auto')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Time (ns)')
    ax.set_title('x–t slice of transient measurements')
    plt.tight_layout()
    return fig
