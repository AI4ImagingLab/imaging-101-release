"""Explicit EPTlib0.1.1 quadratic fit, physical derivatives and Helmholtz law.

Adapted from Alessandro Arduino/INRiM EPTlib, MIT license; see task NOTICE.
General ndimage correlation only applies the locally constructed derivative
stencil. It contains no electric-properties reconstruction implementation.
"""
from functools import lru_cache
import numpy as np
from scipy.ndimage import correlate, binary_erosion

PI = 3.14159265358979323846
MU0 = 4.0e-7 * PI
C0 = 299792458.0
EPS0 = 1.0 / MU0 / C0 / C0
FREQUENCY_HZ = 128.0e6
OMEGA = 2.0 * PI * FREQUENCY_HZ
SPACING_XYZ = (.002, .002, .002)


@lru_cache(maxsize=32)
def kernel_geometry(shape='cube', radius=5):
    """Return read-only bool [z,y,x] footprint; radius is half-width in voxels."""
    if shape not in ('cross', 'sphere', 'cube') or not isinstance(radius, int) or radius < 1:
        raise ValueError('Need cross/sphere/cube and positive integer radius')
    z, y, x = np.indices((2 * radius + 1,) * 3) - radius
    if shape == 'cross':
        inside = (x == 0) & (y == 0) | (x == 0) & (z == 0) | (y == 0) & (z == 0)
    elif shape == 'sphere':
        rho = x.astype(float) * x / radius / radius
        rho += y.astype(float) * y / radius / radius
        rho += z.astype(float) * z / radius / radius
        inside = rho <= 1.0
    else:
        inside = np.ones(x.shape, dtype=bool)
    inside.setflags(write=False)
    return inside


@lru_cache(maxsize=32)
def second_derivative_kernels(shape='cube', radius=5):
    """Return float64 dimensionless coefficients [xyz derivative,z,y,x].

    The symmetric footprint decouples even terms [1,x²,y²,z²] from odd
    linear and mixed-quadratic terms. The source solves the even normal
    equations, then doubles the coefficient of each squared coordinate.
    Physical spacing is applied later by division twice, in source order.
    """
    inside = kernel_geometry(shape, radius)
    z, y, x = np.indices(inside.shape).astype(float) - radius
    design = np.column_stack([np.ones(inside.sum()), x[inside]**2, y[inside]**2, z[inside]**2])
    inverse = np.linalg.inv(design.T @ design) @ design.T
    kernels = np.zeros((3, *inside.shape), dtype=np.float64)
    for axis in range(3):
        kernels[axis, inside] = 2.0 * inverse[1 + axis]
    kernels.setflags(write=False)
    return kernels


def valid_support(mask, shape='cube', radius=5):
    """Fixed support where every active stencil observation exists."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError('Support must be a three-dimensional array')
    return binary_erosion(mask, structure=kernel_geometry(shape, radius), border_value=0)


def laplacian(field, shape='cube', radius=5, spacing=SPACING_XYZ):
    """Return physical Laplacian and fixed mask, both [z,y,x].

    Invalid exterior is excluded from arithmetic and restored as NaN. For
    cross/sphere shapes, inactive bounding-box entries have zero source
    values, rather than zero times a native NaN contaminating valid output.
    """
    field = np.asarray(field)
    if field.ndim != 3 or not np.issubdtype(field.dtype, np.number):
        raise ValueError('Need a numeric three-dimensional field')
    if len(spacing) != 3 or np.any(np.asarray(spacing) <= 0):
        raise ValueError('Spacing must be three positive physical distances')
    mask = valid_support(np.isfinite(field), shape, radius)
    filled = np.where(np.isfinite(field), field, 0.0).astype(np.complex128 if np.iscomplexobj(field) else np.float64)
    result = np.zeros(field.shape, dtype=filled.dtype)
    for axis, kernel in enumerate(second_derivative_kernels(shape, radius)):
        result += correlate(filled, kernel, mode='constant', cval=0.0) / spacing[axis] / spacing[axis]
    result[~mask] = np.nan
    return result, mask


def phase_conductivity(phase, shape='cube', radius=5, frequency=FREQUENCY_HZ, spacing=SPACING_XYZ):
    """Return apparent sigma [S/m] from real transceive phase [radian]."""
    if np.iscomplexobj(phase) or frequency <= 0:
        raise ValueError('Need real phase and positive frequency')
    derivative, mask = laplacian(phase, shape, radius, spacing)
    return derivative / (2.0 * MU0 * (2.0 * PI * frequency)), mask


def full_helmholtz(phase, magnitude, shape='cube', radius=5, frequency=FREQUENCY_HZ, spacing=SPACING_XYZ):
    """Original magnitude-assisted comparison, returning sigma,epsr,mask.

    Magnitude is the author-supplied transmit-sensitivity approximation in
    arbitrary common scale. This comparison is distinct from the selected
    phase-only estimator and is never labeled independently measured truth.
    """
    phase, magnitude = np.asarray(phase), np.asarray(magnitude)
    if phase.shape != magnitude.shape or np.iscomplexobj(phase) or np.iscomplexobj(magnitude):
        raise ValueError('Phase and magnitude must be matching real fields')
    if frequency <= 0 or np.any(magnitude[np.isfinite(magnitude)] <= 0):
        raise ValueError('Need positive magnitude and frequency')
    wave = magnitude * np.exp(.5j * phase)
    derivative, mask = laplacian(wave, shape, radius, spacing)
    omega = 2.0 * PI * frequency
    permittivity = np.full(wave.shape, complex(np.nan, np.nan), dtype=np.complex128)
    np.divide(derivative, -MU0 * omega * omega * wave, out=permittivity, where=mask)
    return -np.imag(permittivity) * omega, np.real(permittivity) / EPS0, mask
