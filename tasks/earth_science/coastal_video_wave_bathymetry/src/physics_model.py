"""Explicit UBathy physical and modal operators; derived from Simarro and Calvete.
AGPL-3.0; source attribution and license are retained in LICENSE/NOTICE.
"""
import numpy as np
from .preprocessing import G

def analytic_signal(values):
    """Analytic temporal signal through an explicit one-sided Fourier multiplier."""
    n = values.shape[-1]
    multiplier = np.zeros(n)
    multiplier[0] = 1
    if n % 2 == 0:
        multiplier[1:n // 2] = 2
        multiplier[n // 2] = 1
    else:
        multiplier[1:(n + 1) // 2] = 2
    return np.fft.ifft(np.fft.fft(values, axis=-1) * multiplier, axis=-1)

def kappa_to_mu(kappa):
    """Native explicit dispersion approximation, with its exact coefficients."""
    kappa = np.clip(np.asarray(kappa), 1e-14, np.inf)
    return kappa / np.sqrt(np.tanh(kappa)) * (
        1 + kappa ** 1.09 * np.exp(
            -(1.56 + 1.28 * kappa + 0.219 * kappa ** 2 + 0.00371 * kappa ** 3)))

def dispersion_wavenumber(omega, depth):
    """Return k [rad/m] for omega [rad/s], depth [m], g=9.80665 m/s^2."""
    if np.any(np.asarray(depth) <= 0):
        raise ValueError("Water depth must be positive")
    return kappa_to_mu(np.asarray(omega) ** 2 * depth / G) / depth

def gamma_from_bed(omega, water_level, bed):
    """Dimensionless omega^2/(g*k(h)); native positive-depth numerical safeguard."""
    depth = np.clip(np.asarray(water_level) - bed, 1e-14, np.inf)
    return np.asarray(omega) ** 2 / (G * dispersion_wavenumber(omega, depth))

def dmd(values, rank=6, dt=1.0):
    """Exact local snapshot SVD/DMD with native inverse cutoff and eigenmode convention."""
    x0, x1 = values[:, :-1], values[:, 1:]
    u, singular, vh = np.linalg.svd(x0, full_matrices=False)
    full_singular = singular.copy()
    u, singular, v = u[:, :rank], singular[:rank], vh.conj().T[:, :rank]
    inverse = np.diag([1 / s if s > 1e-14 else 0 for s in singular])
    reduced = u.conj().T @ (x1 @ (v @ inverse))
    eigenvalues, w = np.linalg.eig(reduced)
    phi = x1 @ (v @ (inverse @ w))
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = 2 * np.pi * dt / np.imag(np.log(eigenvalues))
    return {
        "periods": periods, "phases": np.angle(phi.T), "amplitudes": np.abs(phi.T),
        "eigenvalues": eigenvalues, "spatial_modes": phi, "singular_values": singular,
        "singular_values_full": full_singular,
    }

def phase_difference(phases, reference):
    """Principal phase differences in radians; eliminates an arbitrary modal phase origin."""
    return np.angle(np.exp(1j * (phases - reference)))

def ransac_trial_count(outlier_probability, desired_probability, sample_size=3):
    """Native confidence formula, including numerical probability clips and ceiling."""
    numerator = np.log(np.clip(1 - desired_probability, 1e-12, 1 - 1e-12))
    denominator = np.log(np.clip(
        1 - (1 - outlier_probability) ** sample_size, 1e-12, 1 - 1e-12))
    return int(np.ceil(numerator / denominator))
