"""Forward model for 2D isotropic seismic traveltime tomography.

Solves the isotropic Eikonal equation in 2D Cartesian coordinates:

    (∂T/∂x)² + (∂T/∂z)² = s²(x, z)

where T is the traveltime field and s = 1/v is the slowness.
The solver uses the Fast Marching Method (scikit-fmm), which gives
identical results to the Fast Sweeping Method used in TomoATT for
smooth velocity models.

Reference: Chen et al. (2024), TomoATT paper, Section 2.1 / Section 3.1.
"""

import numpy as np
import skfmm
from scipy.ndimage import map_coordinates


def solve_eikonal(slowness: np.ndarray, dx: float, dz: float,
                  source_x: float, source_z: float) -> np.ndarray:
    """Solve 2D isotropic Eikonal equation from a point source.

    Uses the Fast Marching Method (scikit-fmm).  The source may fall
    anywhere in the domain; the nearest grid node is used as the FMM
    seed, and a small negative bubble is placed there.

    Args:
        slowness: (Nz, Nx) slowness field s(z, x) in s/km.
        dx:       Grid spacing in the x-direction (km).
        dz:       Grid spacing in the z-direction (km).
        source_x: Source x-coordinate (km).
        source_z: Source z-coordinate (km).

    Returns:
        T: (Nz, Nx) traveltime field (seconds).
    """
    Nz, Nx = slowness.shape
    ix = int(round(source_x / dx))
    iz = int(round(source_z / dz))
    ix = np.clip(ix, 0, Nx - 1)
    iz = np.clip(iz, 0, Nz - 1)

    phi = np.ones((Nz, Nx), dtype=np.float64)
    phi[iz, ix] = -1.0

    velocity = 1.0 / slowness.astype(np.float64)
    T = skfmm.travel_time(phi, velocity, dx=[dz, dx])
    return T.astype(np.float32)


def compute_traveltime_at(T: np.ndarray, x: float, z: float,
                          dx: float, dz: float) -> float:
    """Bilinear interpolation of traveltime field at position (x, z).

    Args:
        T:   (Nz, Nx) traveltime field.
        x:   x-coordinate (km).
        z:   z-coordinate (km).
        dx:  Grid spacing in x (km).
        dz:  Grid spacing in z (km).

    Returns:
        Interpolated traveltime (seconds).
    """
    iz_f = z / dz
    ix_f = x / dx
    val = map_coordinates(T, [[iz_f], [ix_f]], order=1, mode='nearest')
    return float(val[0])


def compute_traveltime_gradient(T: np.ndarray,
                                dx: float, dz: float) -> tuple:
    """Compute the spatial gradient of the traveltime field.

    Uses second-order central differences (numpy.gradient).

    Args:
        T:   (Nz, Nx) traveltime field.
        dx:  Grid spacing in x (km).
        dz:  Grid spacing in z (km).

    Returns:
        (dTdz, dTdx): Each (Nz, Nx) array, gradient components (s/km).
    """
    dTdz, dTdx = np.gradient(T.astype(np.float64), dz, dx)
    return dTdz.astype(np.float32), dTdx.astype(np.float32)


def interpolate_gradient(dTdz: np.ndarray, dTdx: np.ndarray,
                         x: float, z: float,
                         dx: float, dz: float) -> np.ndarray:
    """Bilinear interpolation of the traveltime gradient at (x, z).

    Args:
        dTdz: (Nz, Nx) z-gradient of traveltime (s/km).
        dTdx: (Nz, Nx) x-gradient of traveltime (s/km).
        x:    Query x-coordinate (km).
        z:    Query z-coordinate (km).
        dx:   Grid spacing in x (km).
        dz:   Grid spacing in z (km).

    Returns:
        grad: (2,) array [gz, gx] — gradient components (s/km).
    """
    iz_f = z / dz
    ix_f = x / dx
    coords = [[iz_f], [ix_f]]
    gz = float(map_coordinates(dTdz, coords, order=1, mode='nearest')[0])
    gx = float(map_coordinates(dTdx, coords, order=1, mode='nearest')[0])
    return np.array([gz, gx])


def compute_all_traveltimes(slowness: np.ndarray, dx: float, dz: float,
                            sources: np.ndarray,
                            receivers: np.ndarray) -> np.ndarray:
    """Compute synthetic traveltimes for all source–receiver pairs.

    Args:
        slowness:  (Nz, Nx) slowness field (s/km).
        dx:        Grid spacing in x (km).
        dz:        Grid spacing in z (km).
        sources:   (N_src, 2) source positions [[x, z], ...] (km).
        receivers: (N_rec, 2) receiver positions [[x, z], ...] (km).

    Returns:
        T_syn: (N_src, N_rec) synthetic traveltimes (seconds).
    """
    N_src = len(sources)
    N_rec = len(receivers)
    T_syn = np.zeros((N_src, N_rec), dtype=np.float32)

    for n in range(N_src):
        T_n = solve_eikonal(slowness, dx, dz, sources[n, 0], sources[n, 1])
        for m in range(N_rec):
            T_syn[n, m] = compute_traveltime_at(
                T_n, receivers[m, 0], receivers[m, 1], dx, dz)

    return T_syn


def compute_all_traveltime_fields(slowness: np.ndarray, dx: float, dz: float,
                                  sources: np.ndarray) -> np.ndarray:
    """Compute traveltime fields for all sources.

    Args:
        slowness: (Nz, Nx) slowness field (s/km).
        dx:       Grid spacing in x (km).
        dz:       Grid spacing in z (km).
        sources:  (N_src, 2) source positions [[x, z], ...] (km).

    Returns:
        T_fields: (N_src, Nz, Nx) traveltime fields (seconds).
    """
    N_src = len(sources)
    Nz, Nx = slowness.shape
    T_fields = np.zeros((N_src, Nz, Nx), dtype=np.float32)
    for n in range(N_src):
        T_fields[n] = solve_eikonal(
            slowness, dx, dz, sources[n, 0], sources[n, 1])
    return T_fields
