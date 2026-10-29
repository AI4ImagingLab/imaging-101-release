"""Explicit alternating illumination fitting and sparse height recovery."""
import numpy as np
from scipy.sparse.linalg import spsolve
from .preprocessing import decompose
from .physics_model import diffuse_zenith, fit_light, angular_fields, assemble_height_system


def reconstruct(images, angles, mask, specular_mask, smoothing=.1):
    """Return height[H,W] in pixel units and diagnostic arrays.

    Solves the same full-rank least-squares system as the author's sparse QR.
    Local normal equations use SciPy SuperLU and are independently validated
    against that QR on the full bound dataset. Background is explicitly zero.
    """
    rho, phase, intensity = decompose(images, angles, mask)
    light, _, iterations = fit_light(diffuse_zenith(rho), phase, intensity, mask & ~specular_mask)
    theta, adjusted_phase = angular_fields(rho, phase, specular_mask)
    C, d = assemble_height_system(theta, np.minimum(1., intensity), adjusted_phase,
                                 light, mask, specular_mask, smoothing)
    vector = spsolve(C.T@C, C.T@d)
    height = np.zeros(mask.shape)
    height[mask] = vector
    diagnostics = dict(rho=rho, phase=phase, intensity=intensity, theta=theta,
                       adjusted_phase=adjusted_phase, light=light,
                       light_iterations=iterations, residual=C@vector-d,
                       normal_residual=C.T@(C@vector-d), C=C, rhs=d)
    return height, diagnostics
