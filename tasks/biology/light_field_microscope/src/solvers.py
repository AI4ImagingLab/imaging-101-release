"""
Inverse Solvers for the Light-Field Microscope Task.

Implements Richardson-Lucy (RL) deconvolution for the Poisson observation
model of a light-field microscope. The RL update is:

    v^(q+1) = v^(q) * H^T(y / (H v^(q) + eps)) / H^T(1)

where H is the depth-dependent LFM forward projector, H^T is its adjoint
(backward projector), y is the observed light-field image, and v is the
fluorescence volume estimate.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def run_richardson_lucy(
    system,
    observation: np.ndarray,
    iterations: int,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """Run Richardson-Lucy deconvolution for the light-field forward model.

    Parameters
    ----------
    system:
        LFMSystem instance with forward_project and backward_project methods.
    observation:
        2D light-field sensor image (H, W), float64.
    iterations:
        Number of multiplicative RL update steps.
    init:
        Initial volume estimate of shape (H, W, D). Defaults to all-ones.

    Returns
    -------
    np.ndarray
        Non-negative reconstructed volume of shape (H, W, D).
    """
    if init is None:
        init = np.ones(system.tex_shape + (len(system.depths),), dtype=np.float64)

    observation = np.asarray(observation, dtype=np.float64)
    recon = np.asarray(init, dtype=np.float64).copy()

    norm_bp = system.backward_project(np.ones_like(observation))
    norm_bp = np.maximum(norm_bp, _EPS)

    for iteration in range(iterations):
        estimate = system.forward_project(recon)
        ratio = observation / np.maximum(estimate, _EPS)
        correction = system.backward_project(ratio) / norm_bp
        correction[~np.isfinite(correction)] = 0
        correction[correction < 0] = 0
        recon *= correction
        recon[~np.isfinite(recon)] = 0
        recon[recon < 0] = 0
        print(
            f"[RL] iteration {iteration + 1:02d}/{iterations}, "
            f"estimate sum={estimate.sum():.6f}, recon sum={recon.sum():.6f}"
        )

    return recon
