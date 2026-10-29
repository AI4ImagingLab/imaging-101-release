"""Explicit even real SH acquisition representation, BSD-3-Clause adaptation."""
import numpy as np
from scipy.special import sph_harm_y, eval_legendre


def indices(order=8):
    """Return ascending even degree l and ascending m=-l,...,l."""
    pairs = [(l, m) for l in range(0, order + 1, 2) for m in range(-l, l + 1)]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def basis(directions, order=8):
    """Map (...,3) unit directions to (...,45) legacy Descoteaux SH values."""
    directions = np.asarray(directions, dtype=np.float64)
    radius = np.linalg.norm(directions, axis=-1)
    theta = np.arccos(np.clip(directions[..., 2] / radius, -1, 1))
    phi = np.arctan2(directions[..., 1], directions[..., 0])
    l, m = indices(order)
    harmonic = sph_harm_y(l, abs(m), theta[..., None], phi[..., None])
    return np.where(m > 0, harmonic.imag, harmonic.real) * np.where(m == 0, 1., np.sqrt(2.))


def fit_operator(directions, order=8, smooth=.006):
    """Return SH design B and CSA inverse A, preserving augmented SVD fitting."""
    B = basis(directions, order)
    l, _ = indices(order)
    L = -l * (l + 1)
    augmented = np.concatenate([B, np.diag(np.sqrt(smooth) * L)])
    inverse = np.linalg.pinv(augmented)[:, :len(directions)]
    A = (eval_legendre(l, 0) * L / (8 * np.pi))[:, None] * inverse
    return B, A


def sample_odf(coefficients, directions):
    """Evaluate ODF in sr^-1 at supplied unit directions."""
    return coefficients @ basis(directions).T
