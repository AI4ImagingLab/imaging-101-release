"""Closed-form phase-only estimator; no artificial iterative optimizer."""
import numpy as np
from .preprocessing import average_repeats
from .physics_model import phase_conductivity


def reconstruct(phases, shape='cube', radius=5):
    """Return finite sigma float64 [z,y,x], support bool and mean phase.

    Outside the input-defined valid stencil support, the released output
    convention is exactly zero. Signed estimates are preserved inside it.
    """
    mean_phase = average_repeats(phases)
    conductivity, mask = phase_conductivity(mean_phase, shape, radius)
    return np.where(mask, conductivity, 0.0), mask, mean_phase


def reconstruct_repeats(phases, shape='cube', radius=5):
    """Active repeatability comparison [repeat,z,y,x], native NaN exterior."""
    return np.stack([phase_conductivity(sample, shape, radius)[0] for sample in phases])
