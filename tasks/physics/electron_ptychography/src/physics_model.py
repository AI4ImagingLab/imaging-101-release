"""Physics model: ptychographic forward model and center-of-mass computation."""

import numpy as np


def compute_com(datacube, mask=None):
    """Compute center-of-mass of each diffraction pattern.

    The CoM displacement is proportional to the average transverse momentum
    transfer, which equals the gradient of the specimen phase (for thin
    specimens under kinematic scattering).

    Parameters
    ----------
    datacube : np.ndarray, shape (Rx, Ry, Qx, Qy)
        4D-STEM dataset.
    mask : np.ndarray, optional, shape (Qx, Qy)
        Binary mask to restrict CoM computation to the BF disk.

    Returns
    -------
    com_x : np.ndarray, shape (Rx, Ry)
        CoM shift in the Qx direction.
    com_y : np.ndarray, shape (Rx, Ry)
        CoM shift in the Qy direction.
    """
    data = np.asarray(datacube.data if hasattr(datacube, "data") else datacube)
    Rx, Ry, Qx, Qy = data.shape

    qx = np.arange(Qx, dtype=np.float64)
    qy = np.arange(Qy, dtype=np.float64)
    QY, QX = np.meshgrid(qy, qx)

    com_x = np.zeros((Rx, Ry))
    com_y = np.zeros((Rx, Ry))

    for rx in range(Rx):
        for ry in range(Ry):
            dp = np.asarray(data[rx, ry], dtype=np.float64)
            if mask is not None:
                dp = dp * mask
            total = dp.sum()
            if total > 0:
                com_x[rx, ry] = (dp * QX).sum() / total
                com_y[rx, ry] = (dp * QY).sum() / total

    return com_x, com_y


def ptychographic_forward(obj, probe, positions):
    """Ptychographic forward model: compute predicted diffraction intensities.

    For each scan position j, the exit wave is:
        psi_j(r) = P(r - r_j) * O(r)

    The measured intensity is:
        I_j(k) = |F{psi_j(r)}|^2

    Parameters
    ----------
    obj : np.ndarray, complex, shape (Nx, Ny)
        Complex object transmission function.
    probe : np.ndarray, complex, shape (Np, Np)
        Complex probe function.
    positions : np.ndarray, int, shape (J, 2)
        Scan positions (row, col) indexing into the object array.

    Returns
    -------
    intensities : np.ndarray, float, shape (J, Np, Np)
        Predicted diffraction intensities.
    """
    J = positions.shape[0]
    Np = probe.shape[0]
    intensities = np.zeros((J, Np, Np))

    for j in range(J):
        r, c = positions[j]
        obj_patch = obj[r : r + Np, c : c + Np]
        exit_wave = probe * obj_patch
        far_field = np.fft.fft2(exit_wave)
        intensities[j] = np.abs(far_field) ** 2

    return intensities
