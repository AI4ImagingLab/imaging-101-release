"""Forward (A) and transpose (At) operators for snapshot compressive imaging.
Inputs/outputs are in the dispersed detector frame."""
import numpy as np


def A(x, Phi):
    """(H, Wext, nC) * (H, Wext, nC) -> (H, Wext) 2D measurement."""
    return np.sum(x * Phi, axis=2)


def At(y, Phi):
    """(H, Wext) -> (H, Wext, nC) back-projected cube."""
    return np.multiply(np.repeat(y[:, :, np.newaxis], Phi.shape[2], axis=2), Phi)
