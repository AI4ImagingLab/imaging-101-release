"""
solvers.py — f-k migration NLOS reconstruction algorithm.

Implements the wave-based f-k (frequency-wavenumber / Stolt) migration
for confocal NLOS imaging.

Interface:
    result = fk_reconstruction(meas, wall_size, bin_resolution, **kwargs)

where
    meas          : ndarray (Nt, Ny, Nx) — confocal transient measurements
    wall_size     : float — side length of the scanned wall (metres)
    bin_resolution: float — temporal bin width (seconds)
    result        : ndarray (Nt, Ny, Nx) — reconstructed albedo volume
                    dim-0 = depth, dim-1 = y, dim-2 = x

Reference:
  Lindell, Wetzstein, O'Toole, "Wave-Based Non-Line-of-Sight Imaging using
  Fast f-k Migration", ACM Trans. Graph. 38(4), 2019.
  https://github.com/computational-imaging/nlos-fk
"""

import numpy as np
from scipy.ndimage import map_coordinates


def fk_reconstruction(
    meas: np.ndarray,
    wall_size: float,
    bin_resolution: float,
    c: float = 3e8,
) -> np.ndarray:
    """
    Wave-based NLOS reconstruction via fast f-k (Stolt) migration.

    Algorithm (Lindell et al. 2019):
      1. Pre-process: Ψ = sqrt(|meas| · grid_z²)  — amplitude + z² compensation
      2. Pad to 2× in all dimensions.
      3. 3-D FFT + fftshift.
      4. Stolt interpolation: map temporal freq kf → depth freq kz via

             kz_new = sqrt(scale² · (kx² + ky²) + kf²)

         The scale factor is:

             scale = (N · range) / (M · width · 4)

         where range = M · c · bin_resolution, width = wall_size / 2.
         The Jacobian weight |kf| / kz_new accounts for the change of
         variables in the migration.

      5. Inverse 3-D FFT.
      6. Return |result|² on the unpadded half.

    Complexity: O(N³ log N) time, O(N³) memory.

    Parameters
    ----------
    meas : ndarray, shape (Nt, Ny, Nx)
        Confocal transient measurements (photon counts / intensity).
    wall_size : float
        Side length of the scanned wall region in metres.
    bin_resolution : float
        Temporal bin width in seconds.
    c : float
        Speed of light in m/s.

    Returns
    -------
    vol : ndarray, shape (Nt, Ny, Nx), float64
        Reconstructed albedo volume (non-negative).
    """
    M, N, _ = meas.shape
    width   = wall_size / 2.0
    range_m = M * c * bin_resolution
    scale   = (N * range_m) / (M * width * 4.0)

    # ---- Step 0: pre-process ------------------------------------------------
    grid_z = np.linspace(0.0, 1.0, M, dtype=np.float64)[:, None, None]
    data   = np.abs(meas) * grid_z**2
    data   = np.sqrt(data)

    # ---- Pad ----------------------------------------------------------------
    tdata = np.zeros((2 * M, 2 * N, 2 * N), dtype=np.float64)
    tdata[:M, :N, :N] = data

    # ---- Step 1: 3-D FFT ----------------------------------------------------
    tdata = np.fft.fftshift(np.fft.fftn(tdata))

    # ---- Step 2: Stolt interpolation ----------------------------------------
    z_1d = np.arange(-M, M, dtype=np.float64) / M
    y_1d = np.arange(-N, N, dtype=np.float64) / N
    x_1d = np.arange(-N, N, dtype=np.float64) / N
    z3d, y3d, x3d = np.meshgrid(z_1d, y_1d, x_1d, indexing='ij')

    # New temporal-freq coordinate after Stolt mapping
    z_new = np.sqrt(np.abs(scale**2 * (x3d**2 + y3d**2) + z3d**2))

    # Convert to array indices  (z_1d[k] = (k - M)/M  →  k = z·M + M)
    z_arr = z_new * M + M
    y_arr = y3d   * N + N   # y_new == y3d, so these are exact integers
    x_arr = x3d   * N + N

    coords = np.array([z_arr.ravel(), y_arr.ravel(), x_arr.ravel()])
    tr = map_coordinates(tdata.real, coords, order=1, mode='constant', cval=0.0)
    ti = map_coordinates(tdata.imag, coords, order=1, mode='constant', cval=0.0)
    tvol = (tr + 1j * ti).reshape(2 * M, 2 * N, 2 * N)

    # Keep only positive-depth frequencies; apply Jacobian weight
    tvol *= (z3d > 0)
    tvol *= np.abs(z3d) / np.maximum(z_new, 1e-6)

    # ---- Step 3: inverse 3-D FFT --------------------------------------------
    tvol = np.fft.ifftn(np.fft.ifftshift(tvol))
    tvol = np.abs(tvol)**2

    # ---- Unpad and return ---------------------------------------------------
    vol = np.abs(tvol[:M, :N, :N])
    return vol
