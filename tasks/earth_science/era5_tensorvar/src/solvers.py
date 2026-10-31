"""Tensor-Var 4D-Var solver in deep kernel feature space.

The Tensor-Var formulation lifts the state and the observation into a shared
finite-dimensional feature space via two learned encoders, then runs a *linear*
4D-Var by solving a quadratic program in feature space. Once the optimal
feature trajectory is found, the encoder's preimage network decodes it back to
the physical state. This module exposes that pipeline as two pure functions.

Variables (matching the notation in `plan/approach.md`):
    z_b : background mean of the state feature distribution, shape (D,)
    seq_z : observation features for one window, shape (T, D)
    F : linear forward operator in feature space, shape (D, D)
    B, R, Q : information (precision) matrices in feature space, all (D, D)
    T : assimilation window length

The QP minimises::

    J(z_{0:T}) = (z_0 - z_b)^T B (z_0 - z_b)
               + sum_{t=0}^{T-1}     (z_t - phi_o(o_t))^T R (z_t - phi_o(o_t))
               + sum_{t=1}^{T-1} w_t (z_t - F^T z_{t-1})^T Q (z_t - F^T z_{t-1})

with `w_t` a linearly-decreasing penalty across the window.
"""

from __future__ import annotations

import time
from typing import Sequence

import cvxpy as cp
import numpy as np
import torch
from cvxpy import quad_form

from .physics_model import ERA5ForwardModel, ERA5InverseModel


def qp_solver_latent(
    z_b: np.ndarray,
    seq_z: np.ndarray,
    F: np.ndarray,
    B: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    T: int,
) -> np.ndarray:
    """Quadratic program for one assimilation window in feature space.

    Returns the analysed feature trajectory of shape (T, D).
    """
    state_dim = z_b.shape[0]
    pred_seq_z = cp.Variable((T, state_dim))
    cost = 0
    penalty = np.linspace(0.1, 1, T - 1)[::-1]

    for t in range(T):
        if t == 0:
            cost += quad_form(pred_seq_z[t] - z_b, B)
            cost += quad_form(pred_seq_z[t] - seq_z[t], R)
        else:
            cost += quad_form(pred_seq_z[t] - seq_z[t], R)
            cost += quad_form(pred_seq_z[t] - F.T @ pred_seq_z[t - 1], penalty[t - 1] * Q)

    cp.Problem(cp.Minimize(cost), []).solve()
    return pred_seq_z.value


def tensor_var_4dvar(
    obs_history: torch.Tensor,
    forward_model: ERA5ForwardModel,
    inverse_model: ERA5InverseModel,
    z_b: torch.Tensor,
    B: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    assimilation_window: int,
    total_steps: int,
) -> tuple[np.ndarray, dict]:
    """Run Tensor-Var 4D-Var over a single observation sequence.

    Parameters
    ----------
    obs_history : torch.Tensor
        Normalised observation history for the window, shape (T, history_len*C, H, W).
    forward_model, inverse_model : trained nn.Modules
    z_b : torch.Tensor
        Background mean of the state feature, shape (D,).
    B, R, Q : np.ndarray
        Feature-space information matrices, each (D, D).
    assimilation_window : int
        Length of one inner 4D-Var window.
    total_steps : int
        Total number of timesteps to assimilate. Equal to `assimilation_window`
        for the single-window setup used in this task.

    Returns
    -------
    trajectory : np.ndarray
        Decoded analysis trajectory in normalised state space, shape
        (total_steps, C, H, W).
    diagnostics : dict
        Captures the per-window intermediates that the parity tests check.
    """
    device = obs_history.device
    F = forward_model.C_forward.detach().cpu().numpy()
    z_b_np = z_b.detach().cpu().numpy()

    inv_obs_seq = []
    K_S_seq = []
    qp_results = []
    decoded_windows = []

    start_time = time.time()
    with torch.no_grad():
        for start in range(0, total_steps, assimilation_window):
            end = min(start + assimilation_window, total_steps)
            obs_window = obs_history[start:end]

            inv_seq = inverse_model(obs_window)
            inv_obs_seq.append(inv_seq.detach().cpu().numpy())

            seq_z, encode_list = forward_model.encode(inv_seq, return_encode_list=True)
            seq_z_np = seq_z.detach().cpu().numpy()
            K_S_seq.append(seq_z_np)

            qp_result = qp_solver_latent(
                z_b=z_b_np,
                seq_z=seq_z_np,
                F=F,
                B=B,
                R=R,
                Q=Q,
                T=end - start,
            )
            qp_results.append(qp_result)

            decoded = forward_model.decode(
                torch.tensor(qp_result, dtype=torch.float32, device=device),
                encode_list,
            )
            decoded_windows.append(decoded.detach().cpu().numpy())

    elapsed = time.time() - start_time
    trajectory = np.concatenate(decoded_windows, axis=0)
    diagnostics = {
        "inv_obs_seq_z": np.concatenate(inv_obs_seq, axis=0),
        "K_S_seq_z": np.concatenate(K_S_seq, axis=0),
        "qp_result": np.concatenate(qp_results, axis=0),
        "evaluation_time_s": elapsed,
    }
    return trajectory, diagnostics
