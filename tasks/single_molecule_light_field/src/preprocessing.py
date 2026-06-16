"""Preprocessing for Single-Molecule Light Field Microscopy (SMLFM).

Loads 2D localisation data from raw_data.npz and prepares centred arrays
ready for microlens assignment and 3D fitting.

Extracted from: hexSMLFM / PySMLFM (TheLeeLab / Photometrics, Cambridge).
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt


def load_localizations(npz_path: Path, meta: dict) -> npt.NDArray[float]:
    """Load 2D localizations from raw_data.npz.

    Supports two raw_data.npz layouts:
    1. Standardized (N, 8) array already converted to physical units:
        [0] frame
        [1] X (microns)
        [2] Y (microns)
        [3] sigma_X (microns)
        [4] sigma_Y (microns)
        [5] intensity (photons)
        [6] background (photons)
        [7] precision (microns)
    2. Raw PeakFit export with many columns, as bundled in this task.
       In that case the required columns are extracted and converted to
       the standardized (N, 8) representation above.

    Args:
        npz_path: Path to the raw_data.npz file.
        meta: Dictionary loaded from meta_data.json (unused, kept for API compatibility).

    Returns:
        locs_2d_csv (N, 8) float array with columns as described above.
    """
    d = np.load(npz_path)
    raw = d["localizations_2d"].astype(float)

    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D localisation array, got shape {raw.shape}")

    if raw.shape[1] == 8:
        return raw

    if meta.get("csv_format", "").upper() == "PEAKFIT" and raw.shape[1] >= 14:
        pixel_size_sample = _pixel_size_sample(meta)
        locs_2d_csv = np.empty((raw.shape[0], 8), dtype=float)
        locs_2d_csv[:, 0] = raw[:, 0]                        # frame
        locs_2d_csv[:, 1] = raw[:, 9]  * pixel_size_sample   # X (px -> um)
        locs_2d_csv[:, 2] = raw[:, 10] * pixel_size_sample   # Y (px -> um)
        locs_2d_csv[:, 3] = raw[:, 12] * pixel_size_sample   # sigma_X (px -> um)
        locs_2d_csv[:, 4] = raw[:, 12] * pixel_size_sample   # sigma_Y (px -> um)
        locs_2d_csv[:, 5] = raw[:, 8]                        # intensity (photons)
        locs_2d_csv[:, 6] = raw[:, 7]                        # background (photons)
        locs_2d_csv[:, 7] = raw[:, 13] / 1000.0              # precision (nm -> um)
        return locs_2d_csv

    raise ValueError(
        f"Unsupported localisation array shape {raw.shape}. "
        "Expected standardized (N, 8) data or raw PEAKFIT columns."
    )


def center_localizations(locs_2d_csv: npt.NDArray[float]) -> npt.NDArray[float]:
    """Subtract the mean X and Y so the field of view is centred at the origin.

    This places the optical axis at (0, 0), which is required for microlens
    assignment (the MLA lattice is also defined relative to its own centre).

    Args:
        locs_2d_csv: (N, 8) array from load_localizations.

    Returns:
        Copy of input with columns 1 and 2 (X, Y) mean-subtracted.
    """
    out = locs_2d_csv.copy()
    out[:, 1] -= out[:, 1].mean()
    out[:, 2] -= out[:, 2].mean()
    return out


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _pixel_size_sample(meta: dict) -> float:
    """Return pixel size in sample space (microns).

    pixel_size_sample = pixel_size_camera / magnification
    magnification     = (f_tube / f_obj) * (f_mla / f_fourier)
    """
    mag = (meta["focal_length_tube_lens"] / meta["focal_length_obj_lens"]
           * meta["focal_length_mla"] / meta["focal_length_fourier_lens"])
    return meta["pixel_size_camera"] / mag
