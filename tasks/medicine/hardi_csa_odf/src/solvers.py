"""Closed-form CSA reconstruction; no domain solver dependencies."""
import numpy as np
from .preprocessing import normalize
from .physics_model import fit_operator


def reconstruct(signal, bvalues, directions, order=8, smooth=.006):
    """Return float64 (...,45) ODF coefficients in sr^-1."""
    attenuation = normalize(signal, bvalues)[..., bvalues > 50]
    _, operator = fit_operator(directions[bvalues > 50], order, smooth)
    transformed = np.log(-np.log(np.clip(attenuation, .001, .999)))
    coefficients = transformed @ operator.T
    coefficients[..., 0] = .5 / np.sqrt(np.pi)
    return coefficients
