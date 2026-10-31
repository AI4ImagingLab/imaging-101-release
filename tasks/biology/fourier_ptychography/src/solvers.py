"""
Reconstruction solvers for Fourier ptychography (FP).

Provides a standalone implementation of the qNewton solver that exactly
reproduces the PtyLab behavior without requiring the ptylab package.

Algorithm reference
-------------------
qNewton for FPM: Loetgering et al., "Quantitative characterization of
partially coherent sources using ptychographic coherence recovery",
Optica 2023.
"""

import numpy as np
from .utils import fft2c, ifft2c
from .preprocessing import PtyData, PtyState


# ---------------------------------------------------------------------------
# qNewton solver
# ---------------------------------------------------------------------------

def run_qnewton(
    state: PtyState,
    data: PtyData,
    params,
    monitor=None,
    num_iterations: int = 200,
    beta_probe: float = 1.0,
    beta_object: float = 1.0,
    reg_object: float = 1.0,
    reg_probe: float = 1.0,
) -> PtyState:
    """
    Run quasi-Newton (qNewton) FPM reconstruction.

    Algorithm per outer iteration:
    1. Sort positions by NA (bright field first).
    2. For each LED position j:
       a. Extract k-space patch from object spectrum.
       b. Apply pupil → exit surface wave (in k-space, Fraunhofer).
       c. Intensity projection: ESW ← ESW * sqrt(I_meas / I_est).
       d. Back-propagate to update eswUpdate.
       e. quasi-Newton object k-space patch update.
       f. quasi-Newton pupil update.
    3. Compute error metric.
    4. Apply constraints (probe boundary, probe power, CoM, adaptive denoising
       is handled per-position inside the intensity projection step).

    Parameters
    ----------
    state : PtyState
        Current reconstruction state (modified in-place and returned).
    data : PtyData
        Measured data and geometry.
    params : SimpleNamespace
        Algorithm settings from setup_params().
    monitor : SimpleNamespace, optional
        Ignored.
    num_iterations : int
    beta_probe, beta_object : float
        qNewton step sizes.
    reg_object, reg_probe : float
        Quasi-Newton regularization denominators.

    Returns
    -------
    PtyState
    """
    J = len(state.positions)

    # Position order: sorted by distance from k-space center (NA order)
    rows = state.positions[:, 0] - np.mean(state.positions[:, 0])
    cols = state.positions[:, 1] - np.mean(state.positions[:, 1])
    dist = np.sqrt(rows ** 2 + cols ** 2)
    pos_order = np.argsort(dist)

    for loop in range(num_iterations):
        error_at_pos = np.zeros(J)

        for j_idx in pos_order:
            row, col = state.positions[j_idx]
            sy = slice(row, row + state.Np)
            sx = slice(col, col + state.Np)

            # Extract k-space patch (in fftshift convention: DC at center)
            kspace_patch = state.object[sy, sx].copy()

            # Exit wave: k-space patch × pupil
            esw = kspace_patch * state.probe

            # Propagate to detector (Fraunhofer: FPM uses fft2c)
            ESW = fft2c(esw)
            Iest = np.abs(ESW) ** 2
            Imeas = data.ptychogram[j_idx].astype(np.float64)

            # Adaptive denoising (clip noise floor)
            if getattr(params, "adaptiveDenoisingSwitch", False):
                Ameas = np.sqrt(np.abs(Imeas))
                Aest = np.sqrt(np.abs(Iest))
                noise = np.abs(np.mean(Ameas - Aest))
                Ameas = np.maximum(0, Ameas - noise)
                Imeas = Ameas ** 2

            # Error accumulation
            error_at_pos[j_idx] = (
                np.sum(np.abs(Imeas - Iest)) / (data.energy_at_pos[j_idx] + 1e-20)
            )

            # Intensity projection (standard)
            frac = np.sqrt(Imeas / (Iest + 1e-10))
            ESW_new = ESW * frac

            # Back-propagate
            esw_update = ifft2c(ESW_new)
            DELTA = esw_update - esw

            # --- qNewton object k-space patch update ---
            # Pmax = max(sum(|probe|))  (sum over all axes for 6D; here just max of abs)
            Pmax = np.max(np.abs(state.probe))
            absP = np.abs(state.probe)
            frac_O = (absP / (Pmax + 1e-20)) * state.probe.conj() / (absP ** 2 + reg_object)
            kspace_patch = kspace_patch + beta_object * frac_O * DELTA
            state.object[sy, sx] = kspace_patch

            # --- qNewton probe update ---
            # Omax = max(sum(|object|)) (here just max of abs of k-space patch)
            Omax = np.max(np.abs(state.object))
            absO = np.abs(kspace_patch)
            frac_P = (absO / (Omax + 1e-20)) * kspace_patch.conj() / (absO ** 2 + reg_probe)
            state.probe = state.probe + beta_probe * frac_P * DELTA

        # Error metric for this iteration
        state.error.append(float(np.sum(error_at_pos)))

        # Apply constraints
        _apply_constraints_fpm(state, data, params, loop)

    return state


# ---------------------------------------------------------------------------
# Constraint application
# ---------------------------------------------------------------------------

def _apply_constraints_fpm(
    state: PtyState,
    data: PtyData,
    params,
    loop: int,
) -> None:
    """Apply FPM constraints after each outer iteration (in-place)."""

    # Probe power correction
    if getattr(params, "probePowerCorrectionSwitch", False):
        probe_power = np.sqrt(np.sum(np.abs(state.probe) ** 2))
        state.probe = state.probe / (probe_power + 1e-12) * data.max_probe_power

    # Probe boundary: zero outside NA circle
    if getattr(params, "probeBoundary", False):
        state.probe = state.probe * state.probeWindow

    # Center-of-mass stabilization (probe only, FPM)
    com_freq = getattr(params, "comStabilizationSwitch", 0)
    if com_freq and com_freq is not False and (loop % int(com_freq) == 0):
        _com_stabilization_probe(state)


def _com_stabilization_probe(state: PtyState) -> None:
    """Roll probe so its centre-of-mass is at the array centre."""
    P2 = np.abs(state.probe)
    dxp = state.dxo
    denom = np.sum(P2) * dxp + 1e-20
    xc = int(np.round(np.sum(state.Xp * P2) / denom))
    yc = int(np.round(np.sum(state.Yp * P2) / denom))
    if xc ** 2 + yc ** 2 > 1:
        state.probe = np.roll(state.probe, (-yc, -xc), axis=(-2, -1))


# ---------------------------------------------------------------------------
# Error metric utility
# ---------------------------------------------------------------------------

def compute_reconstruction_error(
    ptychogram: np.ndarray, ptychogram_est: np.ndarray
) -> float:
    """
    Normalized amplitude error: ||sqrt(I_meas) - sqrt(I_est)||² / ||sqrt(I_meas)||²
    """
    amp_meas = np.sqrt(np.abs(ptychogram))
    amp_est = np.sqrt(np.abs(ptychogram_est))
    return float(np.sum((amp_meas - amp_est) ** 2) / (np.sum(amp_meas ** 2) + 1e-20))
