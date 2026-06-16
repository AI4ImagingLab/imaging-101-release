"""physics_model.py — ERM velocity, Stolt mapping, steering delays."""

import numpy as np


def erm_velocity(c: float, TXangle: float) -> float:
    """Compute the ERM (exploding reflector model) effective velocity.

    Under the ERM, the round-trip propagation at steering angle theta is
    equivalent to one-way propagation at

        v = c / sqrt(1 + cos(theta) + sin(theta)^2)

    Parameters
    ----------
    c : float
        Speed of sound in tissue (m/s).
    TXangle : float
        Steering angle in radians.

    Returns
    -------
    float
        ERM effective velocity (m/s).
    """
    sinA = np.sin(TXangle)
    cosA = np.cos(TXangle)
    return c / np.sqrt(1.0 + cosA + sinA ** 2)


def stolt_fkz(f: np.ndarray, Kx: np.ndarray,
               c: float, TXangle: float) -> np.ndarray:
    """Compute the Stolt-mapped frequency f_kz.

    The Stolt mapping moves energy from the recorded temporal frequency f
    to the migrated axial frequency:

        f_kz = v * sqrt(Kx^2 + 4*f^2 / (beta^2 * c^2))

    where:
        v    = erm_velocity(c, TXangle)
        beta = (1 + cos(theta))^1.5 / (1 + cos(theta) + sin(theta)^2)

    Parameters
    ----------
    f : np.ndarray, shape (nf, nkx)
        Temporal frequency grid (Hz).
    Kx : np.ndarray, shape (nf, nkx)
        Lateral spatial frequency grid (cycles/m).
    c : float
        Speed of sound (m/s).
    TXangle : float
        Steering angle (rad).

    Returns
    -------
    np.ndarray, shape (nf, nkx)
        Migrated frequency values f_kz (Hz).
    """
    sinA = np.sin(TXangle)
    cosA = np.cos(TXangle)
    v    = erm_velocity(c, TXangle)
    beta = (1.0 + cosA) ** 1.5 / (1.0 + cosA + sinA ** 2)
    return v * np.sqrt(Kx ** 2 + 4.0 * (f ** 2) / (beta ** 2 * c ** 2))


def steering_delay(nx: int, pitch: float, c: float,
                   TXangle: float, t0: float = 0.0) -> np.ndarray:
    """Per-element transmit delay for steering compensation.

    For a steered plane wave, element e fires at time:

        t_shift[e] = sin(theta) * ((nx-1)*(theta<0) - e) * pitch / c + t0

    The sign convention follows the reference: TXangle > 0 means the first
    element fires first (wavefront sweeps from element 0 to element nx-1).

    Parameters
    ----------
    nx : int
        Number of array elements.
    pitch : float
        Element pitch (m).
    c : float
        Speed of sound (m/s).
    TXangle : float
        Steering angle (rad).
    t0 : float
        Acquisition start time (s).

    Returns
    -------
    np.ndarray, shape (nx,)
        Delay in seconds for each element.
    """
    sinA = np.sin(TXangle)
    ref  = (nx - 1) * (TXangle < 0)   # reference element index
    return sinA * (ref - np.arange(nx)) * pitch / c + t0
