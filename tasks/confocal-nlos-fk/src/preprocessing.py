"""
preprocessing.py — Data loading and preprocessing for NLOS measurements.

Handles:
  - Loading measurement data from .npz or .mat files
  - TOF (time-of-flight) alignment based on calibration grid
  - Temporal cropping
  - Dimension permutation between storage format and internal format

Storage format (raw_data.npz):   meas shape (Ny, Nx, Nt)  — matches MATLAB convention
Internal computation format:      meas shape (Nt, Ny, Nx)  — temporal axis first
"""

import numpy as np


def load_nlos_data(path: str) -> dict:
    """
    Load NLOS measurement data from .npz or .mat file.

    Expected keys in the file:
        meas    : ndarray (Ny, Nx, Nt) — transient measurements
        tofgrid : ndarray (Ny, Nx)     — TOF calibration delays in picoseconds
                  (optional; pass None / omit if data is already aligned)

    Parameters
    ----------
    path : str
        Path to .npz or .mat file.

    Returns
    -------
    dict with keys 'meas' (Ny, Nx, Nt) and 'tofgrid' (Ny, Nx) or None.
    """
    if path.endswith('.mat'):
        try:
            from scipy.io import loadmat
            raw = loadmat(path)
            meas    = np.array(raw['meas'],    dtype=np.float64)
            tofgrid = np.array(raw['tofgrid'], dtype=np.float64) if 'tofgrid' in raw else None
        except NotImplementedError:
            # MATLAB v7.3 HDF5 format — use h5py
            import h5py
            with h5py.File(path, 'r') as f:
                # h5py reads MATLAB (Ny,Nx,Nt) stored in Fortran order as (Nt,Nx,Ny)
                # Transpose back to storage convention (Ny,Nx,Nt)
                meas = f['meas'][:].transpose(2, 1, 0).astype(np.float64)
                tofgrid = f['tofgrid'][:].T.astype(np.float64) if 'tofgrid' in f else None
    else:
        raw     = np.load(path)
        meas    = np.array(raw['meas'],    dtype=np.float64)
        tofgrid = np.array(raw['tofgrid'], dtype=np.float64) if 'tofgrid' in raw else None

    return {'meas': meas, 'tofgrid': tofgrid}


def preprocess_measurements(
    meas: np.ndarray,
    tofgrid: np.ndarray | None,
    bin_resolution: float,
    crop: int = 512,
) -> np.ndarray:
    """
    Align and crop confocal transient measurements.

    Steps:
      1. For each scan point (i, j), circularly shift the time axis so that
         t = 0 corresponds to the moment the laser pulse hits the wall.
         The shift amount is derived from the TOF calibration grid (in ps).
      2. Crop to the first `crop` time bins (discards the direct-light peak
         and reduces memory).
      3. Permute from storage format (Ny, Nx, Nt) to internal format (Nt, Ny, Nx).

    Parameters
    ----------
    meas : ndarray, shape (Ny, Nx, Nt)
        Raw transient measurements (storage format).
    tofgrid : ndarray, shape (Ny, Nx), or None
        TOF calibration delays in picoseconds.  Pass None if already aligned.
    bin_resolution : float
        Temporal bin width in seconds.
    crop : int
        Number of time bins to keep after alignment.

    Returns
    -------
    data : ndarray, shape (crop, Ny, Nx), float64
        Aligned and cropped measurements in internal computation format.
    """
    meas = np.array(meas, dtype=np.float64)
    Ny, Nx, Nt = meas.shape

    # TOF alignment: shift each histogram so the direct component is at t = 0
    if tofgrid is not None:
        for i in range(Ny):
            for j in range(Nx):
                shift = -int(np.floor(tofgrid[i, j] / (bin_resolution * 1e12)))
                meas[i, j, :] = np.roll(meas[i, j, :], shift)

    # Temporal crop
    meas = meas[:, :, :crop]

    # Permute: (Ny, Nx, Nt) → (Nt, Ny, Nx)
    data = np.transpose(meas, (2, 0, 1))
    return data


def volume_axes(
    Nt: int,
    Ny: int,
    Nx: int,
    wall_size: float,
    bin_resolution: float,
    c: float = 3e8,
) -> tuple:
    """
    Return physical axis tick values for the reconstructed volume.

    Returns
    -------
    (z_axis, y_axis, x_axis) in metres.
    z_axis : depth  0 … range/2
    y_axis : wall y from -wall_size/2 … +wall_size/2
    x_axis : wall x from -wall_size/2 … +wall_size/2
    """
    range_m = Nt * c * bin_resolution
    z_axis  = np.linspace(0,              range_m / 2,   Nt)
    y_axis  = np.linspace(-wall_size / 2, wall_size / 2, Ny)
    x_axis  = np.linspace(-wall_size / 2, wall_size / 2, Nx)
    return z_axis, y_axis, x_axis
