"""Physics model for X-ray ptychography.

Implements the ptychographic forward model using only numpy and cupy:
  - Patch extraction from the object at scan positions
  - Probe-object multiplication to form exit waves
  - FFT propagation to the detector plane
  - Intensity computation (squared magnitude)

No dependency on tike.
"""

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def _get_xp(arr):
    """Return cupy if arr is a cupy array, else numpy."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def extract_patches(psi, scan, probe_shape):
    """Extract object patches at scan positions.

    Parameters
    ----------
    psi : (..., H, W) complex64
        The object transmission function. For single-slice, shape is (1, H, W).
    scan : (N, 2) float32
        Scan positions as (row, col) pixel coordinates of the patch corner.
    probe_shape : (int, int)
        The (height, width) of each patch to extract.

    Returns
    -------
    patches : (N, ph, pw) complex64
        Extracted object patches.
    """
    xp = _get_xp(psi)
    N = scan.shape[0]
    ph, pw = probe_shape
    # Use the last slice (or only slice) for single-slice ptychography
    obj = psi[-1]
    patches = xp.empty((N, ph, pw), dtype=psi.dtype)
    positions = scan.astype(xp.int32)
    for i in range(N):
        r, c = int(positions[i, 0]), int(positions[i, 1])
        patches[i] = obj[r:r + ph, c:c + pw]
    return patches


def insert_patches(grad_patches, scan, obj_shape, probe_shape):
    """Accumulate patch gradients back into an object-shaped array.

    Parameters
    ----------
    grad_patches : (N, ph, pw) complex64
        Gradient contributions from each scan position.
    scan : (N, 2) float32
        Scan positions.
    obj_shape : (H, W)
        Shape of the object array.
    probe_shape : (int, int)
        The (height, width) of each patch.

    Returns
    -------
    obj_update : (H, W) complex64
        Accumulated gradient in object space.
    """
    xp = _get_xp(grad_patches)
    N = scan.shape[0]
    ph, pw = probe_shape
    obj_update = xp.zeros(obj_shape, dtype=grad_patches.dtype)
    positions = scan.astype(xp.int32)
    for i in range(N):
        r, c = int(positions[i, 0]), int(positions[i, 1])
        obj_update[r:r + ph, c:c + pw] += grad_patches[i]
    return obj_update


def forward(psi, probe, scan):
    """Compute the ptychographic forward model.

    For each scan position:
      1. Extract object patch at that position
      2. Multiply by the probe (exit wave = probe * object_patch)
      3. FFT to detector plane
      4. Compute intensity = |FFT(exit_wave)|^2 summed over probe modes

    Parameters
    ----------
    psi : (D, H, W) complex64
        Object transmission function. D=1 for single-slice.
    probe : (1, 1, S, ph, pw) complex64
        Illumination probe with S incoherent modes.
    scan : (N, 2) float32
        Scan positions in pixel coordinates.

    Returns
    -------
    intensity : (N, ph, pw) float32
        Simulated diffraction pattern intensities.
    """
    xp = _get_xp(psi)
    ph, pw = probe.shape[-2:]
    n_modes = probe.shape[2]

    patches = extract_patches(psi, scan, (ph, pw))  # (N, ph, pw)

    # Exit wave: probe * patch, for each mode
    # probe shape: (1, 1, S, ph, pw) → broadcast with patches (N, ph, pw)
    probe_modes = probe[0, 0]  # (S, ph, pw)

    intensity = xp.zeros((patches.shape[0], ph, pw), dtype=xp.float32)
    for s in range(n_modes):
        exit_wave = probe_modes[s] * patches  # (N, ph, pw)
        farplane = xp.fft.fft2(exit_wave, norm='ortho')  # (N, ph, pw)
        intensity += xp.abs(farplane) ** 2

    return intensity


def simulate_diffraction(probe, psi, scan):
    """Simulate far-field diffraction patterns.

    Convenience wrapper matching the previous API.

    Parameters
    ----------
    probe : (1, 1, S, W, H) complex64
    psi : (D, W', H') complex64
    scan : (N, 2) float32

    Returns
    -------
    data : (N, W, H) float32
    """
    return forward(psi, probe, scan)


def validate_inputs(data, scan, probe, psi):
    """Validate shapes and dtypes of ptychography inputs.

    Parameters
    ----------
    data : (N, W, H) float32
    scan : (N, 2) float32
    probe : (1, 1, S, W, H) complex64
    psi : (D, W', H') complex64

    Raises
    ------
    ValueError
        If any shape or dtype constraint is violated.
    """
    if data.ndim != 3:
        raise ValueError(
            f"data must be 3D (N, W, H), got shape {data.shape}")
    if scan.ndim != 2 or scan.shape[1] != 2:
        raise ValueError(
            f"scan must be (N, 2), got shape {scan.shape}")
    if data.shape[0] != scan.shape[0]:
        raise ValueError(
            f"data and scan position counts differ: "
            f"{data.shape[0]} vs {scan.shape[0]}")
    if probe.ndim != 5 or probe.shape[:2] != (1, 1):
        raise ValueError(
            f"probe must be (1, 1, S, W, H), got shape {probe.shape}")
    if psi.ndim != 3:
        raise ValueError(
            f"psi must be 3D (D, W', H'), got shape {psi.shape}")
    if not np.iscomplexobj(probe):
        raise ValueError("probe must be complex-valued")
    if not np.iscomplexobj(psi):
        raise ValueError("psi must be complex-valued")
