"""Build the 3D CASSI sensing matrix Phi from a 2D coded-aperture mask."""
import numpy as np


def build_phi(mask2d, r=256, c=256, nC=31, step=1):
    Phi = np.zeros((r, c + step * (nC - 1), nC), dtype=np.float64)
    for i in range(nC):
        Phi[:, i:i + c, i] = mask2d
    return Phi
