"""Forward model for the bilinear mixing system D = C S^T.

The hyperspectral imaging (HSI) forward model assumes each pixel's
spectrum is a linear combination of pure component spectra weighted by
their concentrations.
"""

import numpy as np


def forward(C, ST):
    """Compute the modelled data matrix D = C S^T.

    Parameters
    ----------
    C : ndarray, shape (n_pixels, n_components)
        Concentration matrix.
    ST : ndarray, shape (n_components, n_freq)
        Spectral matrix (spectra as rows).

    Returns
    -------
    D : ndarray, shape (n_pixels, n_freq)
    """
    return C @ ST


def residual(C, ST, D_obs):
    """Compute the residual D_obs - C S^T.

    Parameters
    ----------
    C : ndarray, shape (n_pixels, n_components)
        Concentration matrix.
    ST : ndarray, shape (n_components, n_freq)
        Spectral matrix.
    D_obs : ndarray, shape (n_pixels, n_freq)
        Observed data matrix.

    Returns
    -------
    R : ndarray, shape (n_pixels, n_freq)
    """
    return D_obs - forward(C, ST)


def mse(C, ST, D_obs):
    """Mean squared error of the bilinear model.

    Parameters
    ----------
    C : ndarray, shape (n_pixels, n_components)
        Concentration matrix.
    ST : ndarray, shape (n_components, n_freq)
        Spectral matrix.
    D_obs : ndarray, shape (n_pixels, n_freq)
        Observed data matrix.

    Returns
    -------
    err : float
    """
    R = residual(C, ST, D_obs)
    return np.mean(R ** 2)
