"""
Preprocessing for Fourier ptychography (FP).

Loads the PtyLab-format HDF5 dataset and initializes the reconstruction
state without any PtyLab dependency.  All PtyLab classes
(ExperimentalData, Reconstruction, Params, Monitor) are replaced by plain
Python dataclasses and namespace objects.
"""

from __future__ import annotations

import json
import numpy as np
import h5py
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List

from .utils import fft2c, ifft2c, circ


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PtyData:
    """Measured data and experimental geometry for FPM."""
    ptychogram: np.ndarray   # (J, Nd, Nd) float32  — low-resolution images
    encoder: np.ndarray      # (J, 2) float64       — LED positions [m]
    wavelength: float        # [m]
    zled: float              # LED-to-sample distance [m]
    dxd: float               # camera pixel size [m]
    magnification: float
    NA: float
    Nd: int                  # low-resolution image size (pixels)
    No: int                  # high-resolution object size (pixels)
    entrancePupilDiameter: float  # pupil diameter [m]
    energy_at_pos: np.ndarray    # (J,) float — sum of each LR image
    max_probe_power: float       # sqrt(max energy) for power normalization


@dataclass
class PtyState:
    """Reconstruction state for FPM (plain 2D arrays throughout)."""
    object: np.ndarray      # (No, No) complex64 — k-space object (fftshift convention)
    probe: np.ndarray       # (Nd, Nd) complex64 — pupil function
    positions: np.ndarray   # (J, 2) int          — k-space shifts (upper-left of pupil window)
    No: int
    Np: int                 # = Nd
    wavelength: float
    zled: float
    dxo: float              # object/probe pixel size [m] (= dxd / magnification)
    Xp: np.ndarray          # (Np, Np) probe coordinate grid [m]
    Yp: np.ndarray          # (Np, Np) probe coordinate grid [m]
    probeWindow: np.ndarray # (Np, Np) float — circular probe boundary mask
    error: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_experimental_data(data_dir: str | Path = "data") -> PtyData:
    """
    Load an FPM dataset from raw_data.npz and meta_data.json.

    raw_data.npz keys: ptychogram, encoder
    meta_data.json keys: wavelength_m, z_led_m, dxd_m, magnification, NA, No

    Returns
    -------
    PtyData
    """
    data_dir = Path(data_dir)
    npz = np.load(data_dir / "raw_data.npz")
    with open(data_dir / "meta_data.json") as fh:
        meta = json.load(fh)

    ptychogram   = npz["ptychogram"].astype(np.float32)
    encoder      = npz["encoder"].astype(np.float64)
    wavelength   = float(meta["wavelength_m"])
    zled         = float(meta["z_led_m"])
    dxd          = float(meta["dxd_m"])
    magnification = float(meta["magnification"])
    NA           = float(meta["NA"]) if "NA" in meta else None
    No           = int(meta["No"]) if "No" in meta else None
    epd          = float(meta["entrance_pupil_diameter_m"]) if "entrance_pupil_diameter_m" in meta else None

    Nd = int(ptychogram.shape[-1])
    dxp = dxd / magnification

    # per-frame energy and probe power normalization
    energy_at_pos = np.sum(ptychogram, axis=(-1, -2))
    max_probe_power = float(np.sqrt(np.max(energy_at_pos)))

    # fallback: compute No from encoder using same formula as PtyLab
    if No is None:
        conv = -(1.0 / wavelength) * dxp * Nd
        dists = np.sqrt(encoder[:, 0] ** 2 + encoder[:, 1] ** 2 + zled ** 2)
        shifts = np.round(conv * encoder / dists[:, None]).astype(int)
        range_pix = int(np.max(shifts.max(axis=0) - shifts.min(axis=0))) + 2 * Nd
        if range_pix % 2 == 1:
            range_pix += 1
        No = max(Nd, range_pix)

    # fallback: compute NA and entrance pupil diameter
    if NA is None:
        NA = 0.1
    if epd is None:
        epd = 2 * NA * Nd * dxp

    return PtyData(
        ptychogram=ptychogram,
        encoder=encoder,
        wavelength=wavelength,
        zled=zled,
        dxd=dxd,
        magnification=magnification,
        NA=NA,
        Nd=Nd,
        No=No,
        entrancePupilDiameter=epd,
        energy_at_pos=energy_at_pos,
        max_probe_power=max_probe_power,
    )


# ---------------------------------------------------------------------------
# Reconstruction setup
# ---------------------------------------------------------------------------

def setup_reconstruction(data: PtyData, seed: int = 0) -> PtyState:
    """
    Initialize the FPM reconstruction state.

    Probe is initialized as a circular pupil (circ).
    Object is initialized as 'upsampled': low-res estimate padded to No×No.

    Returns
    -------
    PtyState
    """
    wavelength = data.wavelength
    Nd = data.Nd
    Np = Nd
    No = data.No
    dxp = data.dxd / data.magnification
    dxo = dxp

    # Coordinate grids (probe plane)
    xp = np.linspace(-Np / 2, Np / 2, Np) * dxp
    Xp, Yp = np.meshgrid(xp, xp)

    # k-space positions for each LED (upper-left corner of Nd×Nd pupil window)
    # Matches PtyLab Reconstruction.positions for FPM:
    #   conv = -(1/λ) * dxo * Np
    #   positions = round(conv * encoder / dist) + No//2 - Np//2
    conv = -(1.0 / wavelength) * dxo * Np
    dists = np.sqrt(data.encoder[:, 0] ** 2 + data.encoder[:, 1] ** 2 + data.zled ** 2)
    positions = (
        np.round(conv * data.encoder / dists[:, None]) + No // 2 - Np // 2
    ).astype(int)

    # --- Probe initialization: circular aperture ---
    # PtyLab overrides the HDF5 entrancePupilDiameter with this formula
    # (matches Reconstruction.computeParameters for FPM):
    #   epd = 2 * dxp^2 * Np * NA / wavelength
    epd = 2 * dxp ** 2 * Np * data.NA / wavelength
    probe_mask = circ(Xp, Yp, epd).astype(np.complex64)
    rng = np.random.default_rng(seed)
    probe = probe_mask * (
        np.ones((Np, Np), dtype=np.complex64)
        + (0.001 * rng.random((Np, Np))).astype(np.float32)
    )

    # Probe boundary window (for probeBoundary constraint)
    # PtyLab uses epd * 1.2 (20% larger than probe support, from BaseEngine._probeWindow)
    probe_window = circ(Xp, Yp, epd * 1.2).astype(np.float32)

    # --- Object initialization: upsampled low-res estimate ---
    # low_res = ifft2c(sqrt(mean(ptychogram)))  → real-space amplitude, padded to No×No
    low_res = ifft2c(np.sqrt(np.mean(data.ptychogram.astype(np.float64), axis=0)))
    pad_size = (No - Np) // 2
    obj = np.pad(low_res, pad_size, mode="constant", constant_values=0).astype(np.complex64)
    # Ensure shape is exactly (No, No)
    obj = obj[:No, :No]
    obj = np.ones((No, No), dtype=np.complex64) * obj

    return PtyState(
        object=obj,
        probe=probe,
        positions=positions,
        No=No,
        Np=Np,
        wavelength=wavelength,
        zled=data.zled,
        dxo=dxo,
        Xp=Xp,
        Yp=Yp,
        probeWindow=probe_window,
    )


# ---------------------------------------------------------------------------
# Parameter / monitor namespaces (replaces PtyLab Params / Monitor)
# ---------------------------------------------------------------------------

def setup_params() -> SimpleNamespace:
    """Return a namespace with standard FPM reconstruction settings."""
    p = SimpleNamespace()
    p.gpuSwitch = False
    p.positionOrder = "NA"         # sorted by NA (bright field first)
    p.probeBoundary = True          # enforce pupil = 0 outside NA circle
    p.adaptiveDenoisingSwitch = True
    p.probePowerCorrectionSwitch = False
    p.comStabilizationSwitch = False
    return p


def setup_monitor(figure_update_freq: int = 10) -> SimpleNamespace:
    """Return a simple monitor namespace (no display)."""
    m = SimpleNamespace()
    m.figureUpdateFrequency = figure_update_freq
    m.verboseLevel = "low"
    return m


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(state: PtyState, filepath: str | Path) -> None:
    """
    Save FPM reconstruction results to an HDF5 file.

    Schema matches PtyLab's Reconstruction.saveResults() output.
    """
    with h5py.File(str(filepath), "w") as hf:
        # Store in 6D shape for compatibility with PtyLab readers
        obj6d = state.object[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]
        probe6d = state.probe[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]
        hf.create_dataset("object", data=obj6d)
        hf.create_dataset("probe", data=probe6d)
        hf.create_dataset("error", data=np.array(state.error))
        hf.create_dataset("wavelength", data=(state.wavelength,))
        hf.create_dataset("dxp", data=(state.dxo,))
