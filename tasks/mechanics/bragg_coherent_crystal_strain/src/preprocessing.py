"""Real APS ZnO observations; scalar geometry is documented in README.

No upstream reader or reconstruction package is imported at runtime.
"""
from pathlib import Path
import numpy as np
from scipy import fft

SHAPE = (121, 256, 256)
RECONSTRUCTION_SHAPE = (126, 256, 256)
ENERGY_KEV = 9.0
DISTANCE_NM = 2.8e9
PIXEL_NM = 55000.0
DELTA_DEG = 28.488
GAMMA_DEG = 0.0478
ROCKING_STEP_DEG = 0.002
WAVELENGTH_NM = 1.2398419843320026 / ENERGY_KEV


def load_observations(path="data/raw_data.npz"):
    """Return counts (2,121,256,256), int32, and theta (2,121), degrees."""
    with np.load(Path(path), allow_pickle=False) as archive:
        counts = archive["photon_counts"]
        theta = archive["theta"]
    if counts.shape != (1, 2, *SHAPE) or counts.dtype != np.int32:
        raise ValueError("photon_counts must have shape (1,2,121,256,256), int32")
    if theta.shape != (1, 2, SHAPE[0]) or theta.dtype != np.float64:
        raise ValueError("theta must have shape (1,2,121), float64")
    if np.any(counts < 0) or not np.isfinite(theta).all():
        raise ValueError("nonnegative counts and finite angles are required")
    return counts[0], theta[0]


def measured_amplitude(counts):
    """Square-root counts in float32, retaining every observed voxel."""
    return np.sqrt(np.asarray(counts, dtype=np.float32))


def center_diffraction_peak(amplitude):
    """Roll the measured maximum to shape//2; return amplitude and index shift.

    This is the required upstream center_max stage, not a fitted phase ramp.
    NumPy's first C-order maximum resolves ties. No voxel is cropped or lost.
    """
    amplitude = np.asarray(amplitude)
    peak = np.unravel_index(np.argmax(amplitude), amplitude.shape)
    shift = np.asarray(amplitude.shape, dtype=np.int64)//2 - np.asarray(peak)
    return np.roll(amplitude, shift, axis=tuple(range(amplitude.ndim))), shift


def prepare_observations(counts):
    """Source auto threshold, sqrt, even-fast padding, center, float32 storage.

    Return amplitude and metadata containing threshold (counts), per-axis
    padding, and integer peak shift. No alien removal, binning, cropping or
    extra shift is configured. This preserves the public count array.
    """
    counts = np.asarray(counts)
    if counts.ndim != 3 or np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("counts must be a finite nonnegative three-dimensional array")
    positive = counts[counts != 0]
    if positive.size == 0:
        raise ValueError("auto intensity threshold requires positive observations")
    threshold = max(2., .141 * float(positive.mean()) - 3.062)
    amplitude = np.sqrt(np.where(counts <= threshold, 0., counts))
    padding = []
    for size in counts.shape:
        target = fft.next_fast_len(size)
        while target % 2:
            target = fft.next_fast_len(target + 1)
        difference = target-size
        padding.append((difference//2, difference-difference//2))
    padding = tuple(padding)
    amplitude = np.pad(amplitude, padding, mode="constant")
    amplitude, shift = center_diffraction_peak(amplitude)
    return amplitude.astype(np.float32), {
        "intensity_threshold": threshold, "padding": padding, "peak_shift": shift}


def reciprocal_geometry():
    """Return nominal B (3,3), rad/nm per array index, and q (3,), rad/nm.

    Rows are laboratory (outboard, up, downstream); columns are the stored
    (rocking, detector row down, detector column) axes. Geometry follows the
    APS 34-ID-C linearization in bcdi, whose copyright is in validation.md.
    """
    delta, gamma, tilt = np.deg2rad([DELTA_DEG, GAMMA_DEG, ROCKING_STEP_DEG])
    k = 2 * np.pi / WAVELENGTH_NM
    column = -k * PIXEL_NM / DISTANCE_NM * np.array([-np.cos(delta), 0., np.sin(delta)])
    row = k * PIXEL_NM / DISTANCE_NM * np.array(
        [np.sin(delta) * np.sin(gamma), -np.cos(gamma), np.cos(delta) * np.sin(gamma)])
    rock = k * tilt * np.array([1 - np.cos(delta) * np.cos(gamma), 0.,
                              np.sin(delta) * np.cos(gamma)])
    q = k * np.array([np.cos(gamma) * np.sin(delta), np.sin(gamma),
                      np.cos(delta) * np.cos(gamma) - 1])
    return np.column_stack([rock, row, column]), q


def real_space_basis(shape=RECONSTRUCTION_SHAPE):
    """Return oblique real-space voxel basis (3,3), nm per array index."""
    basis, _ = reciprocal_geometry()
    return 2 * np.pi * np.linalg.inv(basis).T @ np.diag(1 / np.asarray(shape))
