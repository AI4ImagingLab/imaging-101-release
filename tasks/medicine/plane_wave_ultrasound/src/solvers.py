"""solvers.py — Stolt f-k migration and coherent compounding."""

import numpy as np
from .physics_model import erm_velocity, stolt_fkz, steering_delay


def _interp_lin(dx: float, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Linear interpolation along columns (axis-0).

    Parameters
    ----------
    dx : float
        Uniform grid spacing (xi_i = i * dx).
    y : np.ndarray, shape (nrows, ncols)
        Values on the uniform grid.
    xi : np.ndarray, shape (nrows, ncols)
        Query positions (same shape as y).

    Returns
    -------
    np.ndarray, shape (nrows, ncols)
        Interpolated values; out-of-range entries are set to 0.
    """
    siz  = y.shape
    yi   = np.zeros(siz, dtype=y.dtype)
    idx  = xi / dx
    oob  = idx > (siz[0] - 2)
    idx[oob] = 0            # will be zeroed afterwards
    idxf = np.floor(idx).astype(np.int32)
    frac = idx - idxf

    # Vectorized over columns
    for k in range(siz[1]):
        idxfk = idxf[:, k]
        frack = frac[:, k]
        yi[:, k] = y[idxfk, k] * (1.0 - frack) + y[idxfk + 1, k] * frack

    yi[oob] = 0.0
    return yi


def fkmig(SIG: np.ndarray, fs: float, pitch: float,
          TXangle: float = 0.0, c: float = 1540.0,
          t0: float = 0.0):
    """Stolt f-k migration for a single plane-wave steering angle.

    Parameters
    ----------
    SIG : np.ndarray, shape (N_t, N_x), real or complex float
        RF signals; rows = time samples, columns = array elements.
    fs : float
        Sampling frequency (Hz).
    pitch : float
        Element pitch (m).
    TXangle : float
        Steering angle (rad). Positive = element 0 fires first.
    c : float
        Speed of sound (m/s).
    t0 : float
        Acquisition start time (s).

    Returns
    -------
    x : np.ndarray, shape (N_x,)
        Lateral positions (m), centred on array midpoint.
    z : np.ndarray, shape (N_t,)
        Depth positions (m).
    migSIG : np.ndarray, shape (N_t, N_x), complex128
        Migrated (focused) complex RF image.
    """
    nt, nx = SIG.shape

    # ---- Zero-padding -------------------------------------------------------
    # Time: pad to 4*nt + ntshift (ntshift accounts for t0 offset)
    ntshift = int(2 * np.ceil(t0 * fs / 2))
    ntFFT   = 4 * nt + ntshift
    # Space: factor 1.5 avoids lateral wrap-around
    nxFFT   = int(2 * np.ceil(1.5 * nx / 2))

    # ---- Frequency grids ----------------------------------------------------
    f0 = np.arange(ntFFT // 2 + 1) * fs / ntFFT                       # one-sided
    kx = np.roll(np.arange(-nxFFT / 2, nxFFT / 2) + 1,
                 int(nxFFT // 2 + 1)) / pitch / nxFFT
    Kx, f = np.meshgrid(kx, f0)                                        # (nf, nkx)

    # ---- Axial FFT (time → frequency) ---------------------------------------
    SIG = np.fft.fft(SIG.astype(np.complex128), n=ntFFT, axis=0)
    SIG = SIG[: ntFFT // 2 + 1, :]                                     # positive freqs

    # ---- Steering delay compensation in (f, x) domain ----------------------
    dt            = steering_delay(nx, pitch, c, TXangle, t0)
    tTrim, fTrim  = np.meshgrid(dt, f0)
    SIG           = SIG * np.exp(-2j * np.pi * tTrim * fTrim)

    # ---- Spatial FFT (x → kx) -----------------------------------------------
    SIG = np.fft.fft(SIG, n=nxFFT, axis=1)

    # ---- Stolt f-k mapping (f → f_kz) --------------------------------------
    fkz = stolt_fkz(f, Kx, c, TXangle)

    # Remove evanescent components (sub-sonic lateral phase velocity)
    SIG[np.abs(f) / (np.abs(Kx) + np.spacing(1)) < c] = 0.0

    # Linear interpolation along the frequency axis for each kx column
    df  = fs / ntFFT
    SIG = (_interp_lin(df, SIG.real, fkz)
           + 1j * _interp_lin(df, SIG.imag, fkz))

    # ---- Obliquity factor ---------------------------------------------------
    SIG = SIG * f / (fkz + np.spacing(1))
    SIG[0, :] = 0.0

    # ---- Axial IFFT (reconstruct two-sided spectrum) ------------------------
    # Match the reference implementation's kx ordering before mirroring back
    # to negative temporal frequencies.
    SIGnegf = np.conj(np.fliplr(np.roll(SIG, -1, axis=1)))
    SIGnegf = SIGnegf[ntFFT // 2 - 1: 0: -1, :]      # negative freqs only
    SIG     = np.concatenate((SIG, SIGnegf), axis=0)
    SIG     = np.fft.ifft(SIG, axis=0)

    # ---- Lateral steering compensation in (z, kx) domain -------------------
    # After the axial IFFT, SIG has shape (ntFFT, nxFFT); rebuild full meshgrid
    sinA  = np.sin(TXangle)
    cosA  = np.cos(TXangle)
    gamma = sinA / (2.0 - cosA)
    dz_arr = -gamma * np.arange(ntFFT) / fs * c / 2.0
    Kx_full, gamma_z = np.meshgrid(kx, dz_arr)   # (ntFFT, nxFFT)
    SIG   = SIG * np.exp(-2j * np.pi * Kx_full * gamma_z)

    # ---- Lateral IFFT (kx → x) ---------------------------------------------
    migSIG = np.fft.ifft(SIG, axis=1)
    migSIG = migSIG[np.arange(nt) + ntshift, :nx]

    # ---- Physical coordinate axes -------------------------------------------
    x = (np.arange(nx) - (nx - 1) / 2.0) * pitch
    z = np.arange(nt) * c / 2.0 / fs

    return x, z, migSIG


def coherent_compound(RF: np.ndarray, fs: float, pitch: float,
                      TXangles, c: float = 1540.0, t0: float = 0.0):
    """Migrate all angles and return coherently compounded image.

    Parameters
    ----------
    RF : np.ndarray, shape (N_t, N_x, N_angles), float
        RF signals for all steering angles.
    fs, pitch, c, t0 : float
        Acquisition parameters.
    TXangles : array-like of float, length N_angles
        Steering angles in radians.

    Returns
    -------
    x : np.ndarray, shape (N_x,)
    z : np.ndarray, shape (N_t,)
    compound : np.ndarray, shape (N_t, N_x), complex128
        Mean of per-angle migrated images.
    """
    n_angles = RF.shape[2]
    compound = None
    for i in range(n_angles):
        x, z, mig = fkmig(RF[:, :, i], fs, pitch,
                           TXangle=float(TXangles[i]), c=c, t0=t0)
        if compound is None:
            compound = mig
        else:
            compound = compound + mig
    compound = compound / n_angles
    return x, z, compound
