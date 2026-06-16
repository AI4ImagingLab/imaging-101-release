"""
Linear spectral mixing forward model for Raman spectroscopy.

The observed spectrum at each voxel is modelled as a linear combination of
pure-component endmember spectra weighted by their fractional abundances:

    y_i = E @ a_i

where E is (n_endmembers, B) and a_i is (n_endmembers,) with a_i >= 0
and sum(a_i) = 1.
"""

import numpy as np


def forward(endmembers: np.ndarray, abundances: np.ndarray) -> np.ndarray:
    """Synthesise mixed spectra from endmembers and abundances.

    Parameters
    ----------
    endmembers : ndarray, shape (K, B)
        K endmember spectra, each of length B.
    abundances : ndarray, shape (N, K)
        Fractional abundances for N pixels.

    Returns
    -------
    ndarray, shape (N, B)
        Predicted mixed spectra.
    """
    return abundances @ endmembers


def residual(observed: np.ndarray, endmembers: np.ndarray,
             abundances: np.ndarray) -> np.ndarray:
    """Per-pixel residual: observed - predicted.

    Parameters
    ----------
    observed    : ndarray, shape (N, B)
    endmembers  : ndarray, shape (K, B)
    abundances  : ndarray, shape (N, K)

    Returns
    -------
    ndarray, shape (N, B)
    """
    return observed - forward(endmembers, abundances)


def reconstruction_error(observed: np.ndarray, endmembers: np.ndarray,
                         abundances: np.ndarray) -> float:
    """Root-mean-square reconstruction error over all pixels and bands.

    Parameters
    ----------
    observed    : ndarray, shape (N, B)
    endmembers  : ndarray, shape (K, B)
    abundances  : ndarray, shape (N, K)

    Returns
    -------
    float
    """
    r = residual(observed, endmembers, abundances)
    return float(np.sqrt(np.mean(r ** 2)))
