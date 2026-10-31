"""Forward and transpose models for snapshot compressive imaging (SCI)."""

import numpy as np


def A(x, Phi):
    """Forward model of SCI: collapse multiple coded frames into a single measurement.

    Parameters
    ----------
    x : ndarray, shape (H, W+step*(nC-1), nC)
        Shifted spectral data cube.
    Phi : ndarray, shape (H, W+step*(nC-1), nC)
        3D sensing matrix (coded aperture masks).

    Returns
    -------
    y : ndarray, shape (H, W+step*(nC-1))
        Compressed 2D measurement.
    """
    return np.sum(x * Phi, axis=2)


def At(y, Phi):
    """Transpose of the forward model.

    Parameters
    ----------
    y : ndarray, shape (H, W)
        2D measurement.
    Phi : ndarray, shape (H, W, nC)
        3D sensing matrix.

    Returns
    -------
    x : ndarray, shape (H, W, nC)
        Back-projected spectral data cube.
    """
    return np.multiply(np.repeat(y[:, :, np.newaxis], Phi.shape[2], axis=2), Phi)


def shift(inputs, step):
    """Apply spectral-dependent spatial shift (dispersion).

    Parameters
    ----------
    inputs : ndarray, shape (H, W, nC)
        Spectral data cube.
    step : int
        Shift step size per spectral channel.

    Returns
    -------
    output : ndarray, shape (H, W+(nC-1)*step, nC)
        Shifted spectral data cube.
    """
    row, col, nC = inputs.shape
    output = np.zeros((row, col + (nC - 1) * step, nC))
    for i in range(nC):
        output[:, i * step:i * step + row, i] = inputs[:, :, i]
    return output


def shift_back(inputs, step):
    """Reverse the spectral-dependent spatial shift.

    Parameters
    ----------
    inputs : ndarray, shape (H, W+(nC-1)*step, nC)
        Shifted spectral data cube.
    step : int
        Shift step size per spectral channel.

    Returns
    -------
    output : ndarray, shape (H, W, nC)
        Unshifted spectral data cube.
    """
    row, col, nC = inputs.shape
    for i in range(nC):
        inputs[:, :, i] = np.roll(inputs[:, :, i], (-1) * step * i, axis=1)
    output = inputs[:, 0:col - step * (nC - 1), :]
    return output
