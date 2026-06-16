"""
Reconstruction solvers for conventional ptychography (CP).

Provides standalone implementations of the mPIE and ePIE solvers that
exactly reproduce the PtyLab behavior without requiring the ptylab package.

Algorithm references
--------------------
mPIE: Maiden et al., "An improved ptychographical phase retrieval
      algorithm for diffractive imaging", Ultramicroscopy 2009.
      Momentum acceleration: Thibault & Guizar-Sicairos, New J. Phys. 2012.
"""

import numpy as np
from .utils import fft2c, ifft2c, smooth_amplitude
from .preprocessing import PtyData, PtyState


# ---------------------------------------------------------------------------
# mPIE solver
# ---------------------------------------------------------------------------

def run_mpie(
    state: PtyState,
    data: PtyData,
    params,
    monitor=None,
    num_iterations: int = 350,
    beta_probe: float = 0.25,
    beta_object: float = 0.25,
    iterations_per_round: int = 50,
    seed: int = 0,
) -> PtyState:
    """
    Run momentum-accelerated PIE (mPIE) to reconstruct the CP object and probe.

    The reconstruction is split into rounds of `iterations_per_round` outer
    iterations each.  Even rounds apply L2 regularization; odd rounds do not.
    This schedule follows the PtyLab exampleReconstructionCPM.py example.

    Algorithm per outer iteration:
    1. Shuffle scan positions (random order after first iteration).
    2. For each position j:
       a. Extract object patch, form exit wave.
       b. Fraunhofer propagation to detector.
       c. Intensity projection: ESW ← ESW * sqrt(I_meas / I_est).
       d. Back-propagate to update eswUpdate.
       e. ePIE object update (with regularization α_O).
       f. ePIE probe update (with regularization α_P).
       g. With probability 5%: apply Nesterov momentum to object and probe.
    3. Compute error metric (mean over all positions).
    4. Apply constraints (probe power, CoM, smoothness, L2 reg).

    Parameters
    ----------
    state : PtyState
        Current reconstruction state (modified in-place and returned).
    data : PtyData
        Measured data and geometry.
    params : SimpleNamespace
        Algorithm settings from setup_params().
    monitor : SimpleNamespace, optional
        Ignored (kept for API compatibility).
    num_iterations : int
        Total number of outer iterations (= num_rounds × iterations_per_round).
    beta_probe, beta_object : float
        ePIE step sizes.
    iterations_per_round : int
        Outer iterations per alternating round.
    seed : int
        Random seed for reproducible position shuffling and momentum triggers (default 0).

    Returns
    -------
    PtyState
        Updated state with `.error` list appended.
    """
    alpha_O = 0.1         # object ePIE regularization
    alpha_P = 0.1         # probe  ePIE regularization
    feedbackM = 0.3       # momentum feedback
    frictionM = 0.7       # momentum friction

    rng = np.random.default_rng(seed)

    # Momentum buffers (persist across rounds, reset at start of reconstruction)
    obj_buffer = state.object.copy()
    probe_buffer = state.probe.copy()
    obj_momentum = np.zeros_like(state.object)
    probe_momentum = np.zeros_like(state.probe)

    num_rounds = max(1, num_iterations // iterations_per_round)
    J = len(state.positions)

    for round_idx in range(1, num_rounds + 1):
        apply_l2reg = (round_idx % 2 == 0)

        for loop in range(iterations_per_round):
            # Position order: sequential on first iteration, random thereafter
            if len(state.error) == 0:
                pos_order = np.arange(J)
            else:
                pos_order = rng.permutation(J)

            error_at_pos = np.zeros(J)

            for j_idx in pos_order:
                row, col = state.positions[j_idx]
                sy = slice(row, row + state.Np)
                sx = slice(col, col + state.Np)

                obj_patch = state.object[sy, sx].copy()

                # Exit surface wave and Fraunhofer propagation
                esw = obj_patch * state.probe
                ESW = fft2c(esw)

                # Intensity estimate and measurement
                Iest = np.abs(ESW) ** 2
                Imeas = data.ptychogram[j_idx].astype(np.float64)

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

                # --- Object patch update ---
                P2 = np.abs(state.probe) ** 2
                Pmax = np.max(P2)
                frac_O = state.probe.conj() / (alpha_O * Pmax + (1 - alpha_O) * P2)
                obj_patch = obj_patch + beta_object * frac_O * DELTA
                state.object[sy, sx] = obj_patch

                # --- Probe update ---
                O2 = np.abs(obj_patch) ** 2
                Omax = np.max(O2)
                frac_P = obj_patch.conj() / (alpha_P * Omax + (1 - alpha_P) * O2)
                state.probe = state.probe + beta_probe * frac_P * DELTA

                # --- Probabilistic momentum update (~5% of positions) ---
                if rng.random() > 0.95:
                    # Object momentum
                    grad_O = obj_buffer - state.object
                    obj_momentum = grad_O + frictionM * obj_momentum
                    state.object = state.object - feedbackM * obj_momentum
                    obj_buffer = state.object.copy()
                    # Probe momentum
                    grad_P = probe_buffer - state.probe
                    probe_momentum = grad_P + frictionM * probe_momentum
                    state.probe = state.probe - feedbackM * probe_momentum
                    probe_buffer = state.probe.copy()

            # Error metric for this outer iteration
            state.error.append(float(np.sum(error_at_pos)))

            # Apply constraints
            _apply_constraints_cp(state, data, params, loop, apply_l2reg)

    return state


# ---------------------------------------------------------------------------
# Constraint application
# ---------------------------------------------------------------------------

def _apply_constraints_cp(
    state: PtyState,
    data: PtyData,
    params,
    loop: int,
    apply_l2reg: bool = False,
) -> None:
    """Apply CP constraints after each outer iteration (in-place)."""

    # Probe power correction
    if params.probePowerCorrectionSwitch:
        probe_power = np.sqrt(np.sum(np.abs(state.probe) ** 2))
        state.probe = state.probe / (probe_power + 1e-12) * data.max_probe_power

    # Center-of-mass stabilization (every comStabilizationSwitch iterations)
    com_freq = getattr(params, "comStabilizationSwitch", 0)
    if com_freq and com_freq is not False and (loop % int(com_freq) == 0):
        _com_stabilization(state)

    # Probe amplitude smoothing
    if getattr(params, "probeSmoothenessSwitch", False):
        state.probe = smooth_amplitude(
            state.probe,
            params.probeSmoothenessWidth,
            params.probeSmoothnessAleph,
        )

    # L2 regularization
    if apply_l2reg:
        aleph_obj = getattr(params, "l2reg_object_aleph", 1e-2)
        aleph_prb = getattr(params, "l2reg_probe_aleph", 1e-2)
        state.object *= 1 - aleph_obj
        state.probe *= 1 - aleph_prb


def _com_stabilization(state: PtyState) -> None:
    """
    Roll probe and object so that the probe centre-of-mass is at the array centre.

    Replicates PtyLab's comStabilization (note: PtyLab uses abs(probe), not abs²).
    """
    P2 = np.abs(state.probe)
    dxp = state.dxo
    denom = np.sum(P2) * dxp + 1e-20
    xc = int(np.round(np.sum(state.Xp * P2) / denom))
    yc = int(np.round(np.sum(state.Yp * P2) / denom))
    if xc ** 2 + yc ** 2 > 1:
        state.probe = np.roll(state.probe, (-yc, -xc), axis=(-2, -1))
        state.object = np.roll(state.object, (-yc, -xc), axis=(-2, -1))


# ---------------------------------------------------------------------------
# Error metric utility
# ---------------------------------------------------------------------------

def compute_reconstruction_error(
    ptychogram: np.ndarray, ptychogram_est: np.ndarray
) -> float:
    """
    Normalized amplitude error:  ||sqrt(I_meas) - sqrt(I_est)||² / ||sqrt(I_meas)||²

    Parameters
    ----------
    ptychogram : ndarray, shape (J, Nd, Nd)
        Measured diffraction intensities.
    ptychogram_est : ndarray, shape (J, Nd, Nd)
        Estimated intensities.

    Returns
    -------
    error : float
    """
    amp_meas = np.sqrt(np.abs(ptychogram))
    amp_est = np.sqrt(np.abs(ptychogram_est))
    return float(np.sum((amp_meas - amp_est) ** 2) / (np.sum(amp_meas ** 2) + 1e-20))
