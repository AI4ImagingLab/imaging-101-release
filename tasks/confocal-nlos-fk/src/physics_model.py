"""
physics_model.py — NLOS forward model and supporting operators.

Implements:
  - nlos_forward_model: direct confocal NLOS forward model via spherical integration
  - define_psf: NLOS blur kernel used by FBP and LCT
  - resampling_operator: change-of-variables operator for LCT (t → t²)

Coordinate conventions (internal, after preprocessing):
  - data shape: (Nt, Ny, Nx)  — temporal axis first
  - Nt = number of time bins (temporal resolution)
  - Ny = Nx = N = number of scan points per wall dimension
  - Physical axes:
      t: time (seconds), t[k] = k * bin_resolution
      y, x: wall scan coordinates, y[j] ∈ [-wall_size/2, wall_size/2]
      z (depth): z[k] ∈ [0, Nt * c * bin_resolution / 2]

References:
  Lindell et al., "Wave-Based Non-Line-of-Sight Imaging using Fast f-k Migration",
  ACM Trans. Graph. 38(4), 2019.
  O'Toole et al., "Confocal Non-Line-of-Sight Imaging Based on the Light Cone Transform",
  Nature 555, 2018.
"""

import numpy as np
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Direct confocal NLOS forward model
# ---------------------------------------------------------------------------

def nlos_forward_model(
    rho: np.ndarray,
    wall_size: float,
    bin_resolution: float,
    n_time_bins: int | None = None,
    c: float = 3e8,
) -> np.ndarray:
    """
    Exact confocal NLOS forward model via discrete spherical integration.

    For each non-zero voxel at (z_k, y_j, x_i) with albedo ρ[k,j,i],
    accumulates its contribution to all confocal scan points on the wall:

        τ(x', y', t) = Σ_{x,y,z} ρ(x,y,z) / r⁴ · δ(round(2r/c/dt) − t)

    where r = sqrt((x'−x)² + (y'−y)² + z²) is the one-way distance and dt
    is `bin_resolution`.

    Parameters
    ----------
    rho : ndarray, shape (Nz, Ny, Nx)
        Albedo/reflectivity volume. dim-0 = depth, dim-1 = y, dim-2 = x.
    wall_size : float
        Side length of the scanned wall in metres.
    bin_resolution : float
        Temporal bin width in seconds.
    n_time_bins : int or None
        Number of output time bins. Defaults to Nz.
    c : float
        Speed of light in m/s.

    Returns
    -------
    meas : ndarray, shape (n_time_bins, Ny, Nx), float64
        Transient measurements (photon-count proxy, non-negative).
    """
    Nz, Ny, Nx = rho.shape
    if n_time_bins is None:
        n_time_bins = Nz

    hw    = wall_size / 2.0
    c_dt  = c * bin_resolution                   # distance per bin (metres)
    dz    = c_dt / 2.0                           # depth resolution (one-way)

    # Physical coords of hidden-volume voxels (depth starts at dz/2, not 0)
    z_vals = (np.arange(Nz, dtype=np.float64) + 0.5) * dz
    y_vals = np.linspace(-hw, hw, Ny)
    x_vals = np.linspace(-hw, hw, Nx)

    # Wall scan points share the same lateral grid
    y_scan = y_vals
    x_scan = x_vals

    meas     = np.zeros((n_time_bins, Ny, Nx), dtype=np.float64)
    meas_1d  = meas.ravel()                      # flat view for add.at

    wy_grid = np.arange(Ny, dtype=np.int32)
    wx_grid = np.arange(Nx, dtype=np.int32)
    # Pre-broadcast index arrays (1, Ny, Nx) — tiled per voxel batch
    wy_bcast = wy_grid[None, :, None]            # (1, Ny, 1)
    wx_bcast = wx_grid[None, None, :]            # (1, 1, Nx)

    for kz in range(Nz):
        if rho[kz].max() == 0:
            continue

        z           = z_vals[kz]
        nz_ky, nz_kx = np.where(rho[kz] > 0)   # sparse voxels in this slice
        n_v         = len(nz_ky)
        alb         = rho[kz][nz_ky, nz_kx]     # (n_v,)

        # Distances (n_v, Ny, Nx)
        dy   = y_scan[None, :, None] - y_vals[nz_ky][:, None, None]
        dx   = x_scan[None, None, :] - x_vals[nz_kx][:, None, None]
        r    = np.sqrt(dy**2 + dx**2 + z**2)

        t_f  = np.round(2.0 * r / c_dt).astype(np.int32)  # (n_v, Ny, Nx)
        valid = (t_f >= 0) & (t_f < n_time_bins)

        contrib = alb[:, None, None] / (r**4)    # (n_v, Ny, Nx)

        # Build flat scatter index: t * Ny*Nx + wy * Nx + wx
        wy_exp  = np.broadcast_to(wy_bcast, (n_v, Ny, Nx))
        wx_exp  = np.broadcast_to(wx_bcast, (n_v, Ny, Nx))
        flat_idx = (t_f[valid].astype(np.int64) * Ny * Nx
                    + wy_exp[valid].astype(np.int64) * Nx
                    + wx_exp[valid].astype(np.int64))
        np.add.at(meas_1d, flat_idx, contrib[valid])

    return meas


# ---------------------------------------------------------------------------
# PSF for FBP / LCT
# ---------------------------------------------------------------------------

def define_psf(N: int, M: int, slope: float) -> np.ndarray:
    """
    NLOS blur kernel used by FBP and LCT.

    Computes the shift-invariant 3-D convolution kernel H whose application
    implements the light-cone transform (LCT) forward operator in the
    change-of-variables domain (u = z²).

    Ported faithfully from the nested function definePsf() in
    cnlos_reconstruction.m.

    Parameters
    ----------
    N : int
        Spatial resolution (number of scan points per wall dimension).
    M : int
        Temporal resolution (number of time bins after cropping).
    slope : float
        width / range = (wall_size/2) / (M * c * bin_resolution).

    Returns
    -------
    psf : ndarray, shape (2*M, 2*N, 2*N), float64
        Normalised, circularly-shifted PSF ready for FFT convolution.
    """
    x = np.linspace(-1.0, 1.0, 2 * N)
    y = np.linspace(-1.0, 1.0, 2 * N)
    z = np.linspace( 0.0, 2.0, 2 * M)
    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing='ij')  # (2M, 2N, 2N)

    # Minimum-distance indicator: selects the "shell" voxels
    val = np.abs((4.0 * slope)**2 * (grid_x**2 + grid_y**2) - grid_z)
    psf = (val == val.min(axis=0, keepdims=True)).astype(np.float64)

    # Normalise by the central column, then by overall ℓ₂ norm
    col_sum = psf[:, N, N].sum()
    if col_sum > 0:
        psf /= col_sum
    nrm = np.linalg.norm(psf)
    if nrm > 0:
        psf /= nrm

    # Circular shift so that DC is at the origin (required before FFT convolution)
    psf = np.roll(psf, N, axis=1)
    psf = np.roll(psf, N, axis=2)
    return psf


# ---------------------------------------------------------------------------
# Resampling operator for LCT
# ---------------------------------------------------------------------------

def resampling_operator(M: int):
    """
    Build the non-uniform resampling matrices for the LCT change-of-variables.

    The light-cone transform requires a variable substitution u = t² (z → z²)
    that converts the spherical Radon transform into a 3-D convolution.
    This function constructs the sparse resampling matrix mtx (forward,
    t → u) and its pseudo-inverse mtxi (backward, u → t).

    Ported faithfully from resamplingOperator() in cnlos_reconstruction.m.

    Parameters
    ----------
    M : int
        Number of time bins.

    Returns
    -------
    mtx  : ndarray, shape (M, M), float64
        Forward resampling matrix (t-space → u-space).
    mtxi : ndarray, shape (M, M), float64
        Backward resampling matrix (u-space → t-space, pseudo-inverse).
    """
    x = np.arange(1, M**2 + 1, dtype=np.float64)          # 1 … M²

    # Quadratic mapping: each u-bin x maps to t-bin ceil(sqrt(x)) (1-indexed)
    t_idx = np.ceil(np.sqrt(x)).astype(int) - 1            # 0-indexed, clipped
    t_idx = np.clip(t_idx, 0, M - 1)

    row_idx = np.arange(M**2)
    weights = 1.0 / np.sqrt(x)                             # Jacobian 1/√u

    # Build (M² × M) sparse matrix
    mtx_big = csr_matrix((weights, (row_idx, t_idx)), shape=(M**2, M))
    mtxi_big = mtx_big.T

    mtx  = mtx_big.toarray()
    mtxi = mtxi_big.toarray()

    # Iterative halving: M² → M² / 2 → … → M  (log₂ M steps)
    K = int(round(np.log2(M)))
    for _ in range(K):
        mtx  = 0.5 * (mtx [0::2, :] + mtx [1::2, :])
        mtxi = 0.5 * (mtxi[:, 0::2] + mtxi[:, 1::2])

    return mtx, mtxi
