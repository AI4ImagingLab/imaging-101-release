"""
Preprocessing for conventional ptychography (CP).

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
    """Measured data and experimental geometry for CP."""
    ptychogram: np.ndarray        # (J, Nd, Nd) float32 — diffraction intensities
    encoder: np.ndarray           # (J, 2) float64      — scan positions [m]
    wavelength: float             # [m]
    zo: float                     # sample–detector distance [m]
    dxd: float                    # detector pixel size [m]
    Nd: int                       # detector size (pixels)
    No: int                       # object array size (pixels)
    entrancePupilDiameter: float  # probe beam diameter [m]
    energy_at_pos: np.ndarray     # (J,) float — sum of each diffraction frame
    max_probe_power: float        # sqrt(max energy) for power normalization


@dataclass
class PtyState:
    """Reconstruction state for CP (all arrays are plain 2D numpy arrays)."""
    object: np.ndarray      # (No, No) complex64 — current object estimate
    probe: np.ndarray       # (Np, Np) complex64 — current probe estimate
    positions: np.ndarray   # (J, 2) int          — scan positions in pixels
    No: int
    Np: int
    wavelength: float
    zo: float
    dxo: float              # object/probe pixel size [m]
    Xp: np.ndarray          # (Np, Np) float — probe coord grid [m]
    Yp: np.ndarray          # (Np, Np) float — probe coord grid [m]
    error: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_experimental_data(data_dir: str | Path = "data") -> PtyData:
    """
    Load a CP dataset from raw_data.npz and meta_data.json.

    raw_data.npz keys: ptychogram, encoder
    meta_data.json keys: wavelength_m, zo_m, dxd_m, No, entrance_pupil_diameter_m

    Returns
    -------
    PtyData
    """
    data_dir = Path(data_dir)
    npz = np.load(data_dir / "raw_data.npz")
    with open(data_dir / "meta_data.json") as fh:
        meta = json.load(fh)

    ptychogram = npz["ptychogram"].astype(np.float32)
    encoder    = npz["encoder"].astype(np.float64)
    wavelength = float(meta["wavelength_m"])
    zo         = float(meta["zo_m"])
    dxd        = float(meta["dxd_m"])
    Nd         = int(ptychogram.shape[-1])
    No         = int(meta["No"]) if "No" in meta else None
    epd        = float(meta["entrance_pupil_diameter_m"]) if "entrance_pupil_diameter_m" in meta else None

    # per-frame energy and max probe power (matches PtyLab ExperimentalData)
    energy_at_pos = np.sum(ptychogram, axis=(-1, -2))          # (J,)
    max_probe_power = float(np.sqrt(np.max(energy_at_pos)))

    # fallback: estimate object size if not in file
    if No is None:
        dxp = wavelength * zo / (Nd * dxd)
        positions_raw = np.round(encoder / dxp).astype(int)
        rng = np.max(positions_raw, axis=0) - np.min(positions_raw, axis=0)
        range_pix = int(np.max(rng)) + 2 * Nd
        if range_pix % 2 == 1:
            range_pix += 1
        No = max(Nd, range_pix)

    # fallback: estimate entrance pupil diameter
    if epd is None:
        dxp = wavelength * zo / (Nd * dxd)
        Lp = Nd * dxp
        epd = Lp / 3

    return PtyData(
        ptychogram=ptychogram,
        encoder=encoder,
        wavelength=wavelength,
        zo=zo,
        dxd=dxd,
        Nd=Nd,
        No=No,
        entrancePupilDiameter=epd,
        energy_at_pos=energy_at_pos,
        max_probe_power=max_probe_power,
    )


# ---------------------------------------------------------------------------
# Reconstruction setup
# ---------------------------------------------------------------------------

def setup_reconstruction(data: PtyData, no_scale: float = 1.0, seed: int = 0) -> PtyState:
    """
    Initialize the CP reconstruction state.

    Probe is initialized as a circular aperture with a small focusing
    quadratic phase (matches PtyLab's exampleReconstructionCPM.py).
    Object is initialized as uniform amplitude + tiny random noise.

    Parameters
    ----------
    data : PtyData
    no_scale : float
        Scale factor for the object size (keep at 1.0).
    seed : int
        Random seed for reproducible initialization noise (default 0).

    Returns
    -------
    PtyState
    """
    wavelength = data.wavelength
    zo = data.zo
    Nd = data.Nd
    dxd = data.dxd

    # Pixel size (Fraunhofer CPM formula)
    dxp = wavelength * zo / (Nd * dxd)
    dxo = dxp
    Np = Nd
    No = int(data.No * no_scale)
    if No % 2 == 1:
        No += 1

    # Coordinate grids
    xp = np.linspace(-Np / 2, Np / 2, Np) * dxp
    Xp, Yp = np.meshgrid(xp, xp)

    # Scan positions in pixel coordinates (upper-left corner of probe window)
    positions = (np.round(data.encoder / dxo) + No // 2 - Np // 2).astype(int)

    rng = np.random.default_rng(seed)

    # --- Probe initialization: circular aperture + quadratic phase ---
    probe_mask = circ(Xp, Yp, data.entrancePupilDiameter).astype(np.complex64)
    probe = probe_mask * (
        np.ones((Np, Np), dtype=np.complex64)
        + rng.uniform(0, 0.001, (Np, Np)).astype(np.float32)
    )
    # Add focusing quadratic phase (half-wavelength focus at zo/2)
    probe = probe * np.exp(
        1j * 2 * np.pi / (wavelength * zo * 2) * (Xp ** 2 + Yp ** 2) / 2
    ).astype(np.complex64)

    # --- Object initialization: ones + tiny noise ---
    obj = (
        np.ones((No, No), dtype=np.complex64)
        + rng.uniform(0, 0.001, (No, No)).astype(np.float32)
    )

    return PtyState(
        object=obj,
        probe=probe,
        positions=positions,
        No=No,
        Np=Np,
        wavelength=wavelength,
        zo=zo,
        dxo=dxo,
        Xp=Xp,
        Yp=Yp,
    )


# ---------------------------------------------------------------------------
# Parameter / monitor namespaces (replaces PtyLab Params / Monitor)
# ---------------------------------------------------------------------------

def setup_params() -> SimpleNamespace:
    """Return a namespace with standard CP reconstruction settings."""
    p = SimpleNamespace()
    p.propagatorType = "Fraunhofer"
    p.positionOrder = "random"
    p.gpuSwitch = False
    p.fftshiftSwitch = False

    # probe regularization
    p.probeSmoothenessSwitch = True
    p.probeSmoothnessAleph = 1e-2
    p.probeSmoothenessWidth = 10
    p.comStabilizationSwitch = 10      # run every 10 outer iterations
    p.probePowerCorrectionSwitch = True

    # object regularization (l2reg toggled per round in main.py)
    p.l2reg = False
    p.l2reg_probe_aleph = 1e-2
    p.l2reg_object_aleph = 1e-2

    # unused switches (kept for API compatibility)
    p.TV_autofocus = False
    p.positionCorrectionSwitch = False
    p.adaptiveDenoisingSwitch = False
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
    Save reconstruction results to an HDF5 file.

    Schema matches PtyLab's Reconstruction.saveResults() output.
    """
    with h5py.File(str(filepath), "w") as hf:
        # Store in 6D shape for compatibility with PtyLab readers
        obj6d = state.object[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]
        probe6d = state.probe[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]
        hf.create_dataset("object", data=obj6d)
        hf.create_dataset("probe", data=probe6d)
        hf.create_dataset("error", data=np.array(state.error))
        hf.create_dataset("zo", data=(state.zo,))
        hf.create_dataset("wavelength", data=(state.wavelength,))
        hf.create_dataset("dxp", data=(state.dxo,))
