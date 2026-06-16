"""Synthetic data generation for 2D isotropic seismic traveltime tomography.

Generation pipeline
-------------------
1. **True velocity model** — a checkerboard velocity field is constructed by
   superimposing staggered ±5 % elliptical Gaussian anomalies (4 blocks in x,
   3 blocks in z) on a 1D linear gradient background.  The domain spans
   220 km × 50 km with 2 km grid spacing (111 × 26 nodes).  This synthetic
   experiment follows the design of Fig. 5 in Chen et al. (2024) (TomoATT
   paper, arXiv:2412.00031), adapted to a continental-crust velocity range.

2. **Background velocity model** — a 1D linear gradient
   v(z) = 6.0 + 2.0 × z / 50 km/s (from 6.0 km/s at the surface to 8.0 km/s
   at 50 km depth), representative of consolidated continental crust with the
   Moho near 50 km depth.  This is the starting model for the ATT inversion.

3. **Source / receiver geometry** — 8 seismic stations are placed uniformly on
   the surface (z = 0).  1000 earthquake sources are randomly distributed
   throughout the domain with higher density in the upper-left quadrant
   (x < 0.5 × x_max, z < 0.5 × z_max), mimicking uneven seismicity as in
   the paper.

4. **Synthetic traveltimes** — the Eikonal equation is solved in the true
   velocity model for every source, and traveltimes are extracted at all
   receivers.  Gaussian noise with standard deviation σ = 0.1 s is added to
   mimic pick uncertainty.
"""

import json
import numpy as np
from pathlib import Path

from .physics_model import compute_all_traveltimes


# --------------------------------------------------------------------------- #
#  Default simulation constants                                                #
# --------------------------------------------------------------------------- #

DEFAULT_NX     = 111      # grid points in x  (220 km / 2 km + 1)
DEFAULT_NZ     = 26       # grid points in z  (50 km / 2 km + 1)
DEFAULT_DX_KM  = 2.0      # km
DEFAULT_DZ_KM  = 2.0      # km
DEFAULT_V0     = 6.0      # km/s at z = 0  (upper continental crust)
DEFAULT_V1     = 8.0      # km/s at z = z_max  (lower crust / Moho)
DEFAULT_PERT   = 0.05     # ±5% checkerboard perturbation amplitude
DEFAULT_N_X    = 4        # checkerboard blocks in x
DEFAULT_N_Z    = 3        # checkerboard blocks in z
DEFAULT_N_SRC  = 1000     # earthquake sources
DEFAULT_N_REC  = 8        # surface receivers
DEFAULT_NOISE  = 0.1      # seconds


# --------------------------------------------------------------------------- #
#  Velocity model builders                                                     #
# --------------------------------------------------------------------------- #

def make_background_velocity(Nx: int, Nz: int,
                              dx: float, dz: float,
                              v0: float = DEFAULT_V0,
                              v1: float = DEFAULT_V1) -> np.ndarray:
    """Linear velocity gradient: v(z) = v0 + (v1 - v0) * z / z_max.

    Args:
        Nx:  Number of grid points in x.
        Nz:  Number of grid points in z.
        dx:  Grid spacing in x (km).
        dz:  Grid spacing in z (km).
        v0:  Velocity at z = 0 (km/s).
        v1:  Velocity at z = (Nz-1)*dz (km/s).

    Returns:
        v: (Nz, Nx) velocity array (km/s).
    """
    z_max = (Nz - 1) * dz
    z_vec = np.linspace(0.0, z_max, Nz)
    v_col = v0 + (v1 - v0) * z_vec / z_max
    return np.tile(v_col[:, None], (1, Nx)).astype(np.float32)


def make_marmousi_velocity(data_dir: str) -> np.ndarray:
    """Load the pre-processed Marmousi velocity model (legacy, not used in main pipeline).

    The file ``data_dir/marmousi_velocity.npy`` contains the Marmousi
    model subsampled to (Nz=31, Nx=93) at 2 km (crustal-scale) spacing,
    converted to km/s.  Velocity range: 1.50–5.50 km/s.

    Args:
        data_dir: Path to the task's data/ directory.

    Returns:
        v_marmousi: (Nz=31, Nx=93) float64 velocity array (km/s).
    """
    path = Path(data_dir) / 'marmousi_velocity.npy'
    return np.load(str(path)).astype(np.float64)


def make_checkerboard_perturbation(Nx: int, Nz: int,
                                   dx: float, dz: float,
                                   n_x: int = DEFAULT_N_X,
                                   n_z: int = DEFAULT_N_Z,
                                   pert: float = DEFAULT_PERT,
                                   sigma_frac_x: float = 0.30,
                                   sigma_frac_z: float = 0.45) -> np.ndarray:
    """Elliptical Gaussian checkerboard velocity perturbation (relative, dimensionless).

    Places alternating ±pert elliptical Gaussian anomalies at the centre of
    each n_x × n_z block.  Neighbouring blocks have opposite sign, forming a
    checkerboard pattern.  The Gaussian widths are set as fixed fractions of
    the block dimensions, giving horizontally elongated (2:1 aspect-ratio)
    ellipses that are more representative of crustal velocity anomalies than
    sharp-edged rectangular blocks.

    Args:
        Nx, Nz:       Grid dimensions.
        dx, dz:       Grid spacings (km).
        n_x:          Number of checkerboard columns (default 4).
        n_z:          Number of checkerboard rows    (default 3).
        pert:         Fractional perturbation amplitude (default 0.05 → ±5%).
        sigma_frac_x: σ_x as a fraction of block width  (default 0.30).
        sigma_frac_z: σ_z as a fraction of block height (default 0.45).
                      With block_w ≈ 55 km and block_h ≈ 16.7 km the defaults
                      give σ_x ≈ 16.5 km, σ_z ≈ 7.5 km (aspect ratio ≈ 2.2:1).

    Returns:
        delta_v: (Nz, Nx) relative perturbation (dimensionless), normalised so
                 that max(|delta_v|) == pert.
    """
    x = np.arange(Nx) * dx
    z = np.arange(Nz) * dz
    xx, zz = np.meshgrid(x, z)

    x_max   = (Nx - 1) * dx
    z_max   = (Nz - 1) * dz
    block_w = x_max / n_x
    block_h = z_max / n_z
    sigma_x = block_w * sigma_frac_x
    sigma_z = block_h * sigma_frac_z

    delta_v = np.zeros((Nz, Nx), dtype=np.float64)
    for i in range(n_x):
        for j in range(n_z):
            cx   = (i + 0.5) * block_w
            cz   = (j + 0.5) * block_h
            sign = 1 if (i + j) % 2 == 0 else -1
            gauss = np.exp(
                -((xx - cx) ** 2 / (2.0 * sigma_x ** 2)
                + (zz - cz) ** 2 / (2.0 * sigma_z ** 2))
            )
            delta_v += sign * gauss

    # Normalise: max absolute value = pert
    peak = np.abs(delta_v).max()
    if peak > 0:
        delta_v = delta_v / peak * pert
    return delta_v.astype(np.float32)


def make_true_velocity(Nx: int, Nz: int, dx: float, dz: float,
                       v0: float = DEFAULT_V0, v1: float = DEFAULT_V1,
                       pert: float = DEFAULT_PERT,
                       n_x: int = DEFAULT_N_X,
                       n_z: int = DEFAULT_N_Z) -> np.ndarray:
    """True velocity = background × (1 + checkerboard perturbation).

    Args:
        Nx, Nz: Grid dimensions.
        dx, dz: Grid spacings (km).
        v0:     Background velocity at z=0 (km/s).
        v1:     Background velocity at z=z_max (km/s).
        pert:   Checkerboard amplitude (fractional).
        n_x:    Checkerboard blocks in x.
        n_z:    Checkerboard blocks in z.

    Returns:
        v_true: (Nz, Nx) velocity (km/s).
    """
    v_bg  = make_background_velocity(Nx, Nz, dx, dz, v0, v1)
    delta = make_checkerboard_perturbation(Nx, Nz, dx, dz, n_x, n_z, pert)
    return (v_bg * (1.0 + delta)).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Source / receiver geometry                                                  #
# --------------------------------------------------------------------------- #

def make_receivers(n_rec: int, x_max: float,
                   z_max: float = 0.0,
                   n_surf: int = None,
                   n_side: int = None) -> np.ndarray:
    """Receiver array: surface-only or perimeter layout.

    When z_max == 0 (default), all stations are placed uniformly on the
    surface (z = 0).  When z_max > 0 (perimeter layout), stations are split
    equally among surface (z=0), left boundary (x=0), and right boundary
    (x=x_max).

    Args:
        n_rec:  Total number of receivers.
        x_max:  Domain width in x (km).
        z_max:  Domain depth in z (km).  If 0, surface-only layout.
        n_surf: Number of surface stations (perimeter mode, default: n_rec // 3).
        n_side: Number of stations per side boundary (perimeter mode,
                default: n_rec // 3).

    Returns:
        receivers: (n_rec, 2) positions [[x, z], ...] (km).
    """
    if z_max == 0.0:
        # Surface-only layout
        x_rec = np.linspace(x_max / (n_rec + 1), x_max * n_rec / (n_rec + 1), n_rec)
        return np.column_stack([x_rec, np.zeros(n_rec)]).astype(np.float32)

    # Perimeter layout
    if n_surf is None:
        n_surf = n_rec // 3
    if n_side is None:
        n_side = n_rec // 3

    xs = np.linspace(x_max / (n_surf + 1), x_max * n_surf / (n_surf + 1), n_surf)
    surf = np.column_stack([xs, np.zeros(n_surf)])

    zs_side = np.linspace(z_max / (n_side + 1), z_max * n_side / (n_side + 1), n_side)
    left  = np.column_stack([np.zeros(n_side),        zs_side])
    right = np.column_stack([np.full(n_side, x_max),  zs_side])

    return np.vstack([surf, left, right]).astype(np.float32)


def make_sources(n_src: int, x_max: float,
                 z_max: float = None,
                 depth_levels: tuple = None,
                 seed: int = 42) -> np.ndarray:
    """Earthquake sources with upper-left density bias.

    When z_max is provided (2-D random mode), sources are randomly distributed
    throughout the domain (x, z) with higher density in the upper-left quadrant
    (x < 0.5 * x_max, z < 0.5 * z_max), following the TomoATT checkerboard
    experiment design.

    When depth_levels is provided (legacy fixed-depth mode), sources are placed
    at the given discrete depths with non-uniform x distribution.

    Args:
        n_src:        Total number of sources.
        x_max:        Domain width in x (km).
        z_max:        Domain depth in z (km).  If given, enables 2-D random mode.
        depth_levels: Discrete depth levels (legacy; ignored when z_max is given).
        seed:         Random seed.

    Returns:
        sources: (n_src, 2) positions [[x, z], ...] (km).
    """
    rng = np.random.default_rng(seed)

    if z_max is not None:
        # 2-D random distribution with upper-left density bias
        x_left  = rng.uniform(0.05 * x_max, 0.50 * x_max, n_src)
        x_right = rng.uniform(0.50 * x_max, 0.95 * x_max, n_src)
        x_vals  = np.where(rng.random(n_src) < 0.65, x_left, x_right)

        z_upper = rng.uniform(2.0, 0.50 * z_max, n_src)
        z_lower = rng.uniform(0.50 * z_max, 0.90 * z_max, n_src)
        z_vals  = np.where(rng.random(n_src) < 0.65, z_upper, z_lower)

        return np.column_stack([x_vals, z_vals]).astype(np.float32)

    # Legacy fixed-depth-level mode
    if depth_levels is None:
        depth_levels = (0.0, 10.0, 20.0, 50.0)

    n_depths = len(depth_levels)
    per_depth = n_src // n_depths
    counts = [per_depth] * n_depths
    counts[-1] += n_src - per_depth * n_depths

    sources = []
    for depth, count in zip(depth_levels, counts):
        x_vals = np.where(
            rng.random(count) < 0.65,
            rng.uniform(0.05 * x_max, 0.67 * x_max, count),
            rng.uniform(0.67 * x_max, 0.95 * x_max, count),
        )
        z_vals = np.full(count, depth)
        sources.append(np.column_stack([x_vals, z_vals]))

    return np.vstack(sources).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Main data generation                                                        #
# --------------------------------------------------------------------------- #

def generate_data(output_dir: str = 'data', seed: int = 42) -> dict:
    """End-to-end synthetic data generation.  Saves npz/json under output_dir.

    Builds a checkerboard true velocity model, generates sources/receivers,
    computes synthetic traveltimes with Eikonal forward solver, and saves all
    files.  See module docstring for full pipeline description.

    Args:
        output_dir: Directory to write data files.
        seed:       Random seed for noise and source placement.

    Returns:
        Dictionary with keys 'velocity_true', 'traveltime_obs',
        'sources', 'receivers', 'meta'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Nx    = DEFAULT_NX
    Nz    = DEFAULT_NZ
    dx    = DEFAULT_DX_KM
    dz    = DEFAULT_DZ_KM
    n_src = DEFAULT_N_SRC
    n_rec = DEFAULT_N_REC
    noise = DEFAULT_NOISE
    x_max = (Nx - 1) * dx   # 220 km
    z_max = (Nz - 1) * dz   # 50 km

    print("Building checkerboard velocity model...")
    v_true = make_true_velocity(Nx, Nz, dx, dz,
                                v0=DEFAULT_V0, v1=DEFAULT_V1,
                                pert=DEFAULT_PERT,
                                n_x=DEFAULT_N_X, n_z=DEFAULT_N_Z)
    s_true = (1.0 / v_true).astype(np.float64)

    print("Building source/receiver geometry...")
    receivers = make_receivers(n_rec, x_max)                   # 8 surface stations
    sources   = make_sources(n_src, x_max, z_max=z_max, seed=seed)  # 1000 random 2D

    print(f"Computing synthetic traveltimes ({n_src} sources × {n_rec} receivers)...")
    T_clean = compute_all_traveltimes(s_true, dx, dz, sources, receivers)

    print("Adding Gaussian noise...")
    rng = np.random.default_rng(seed + 1)
    T_obs = (T_clean + rng.normal(0.0, noise, T_clean.shape)).astype(np.float32)

    # ── Save files ────────────────────────────────────────────────────────── #
    np.savez(
        output_dir / 'raw_data.npz',
        traveltime_obs=T_obs[np.newaxis],
        sources=sources[np.newaxis],
        receivers=receivers[np.newaxis],
    )

    np.savez(
        output_dir / 'ground_truth.npz',
        velocity=v_true[np.newaxis],
    )

    meta = {
        'Nx': int(Nx),
        'Nz': int(Nz),
        'dx_km': float(dx),
        'dz_km': float(dz),
        'x_min_km': 0.0,
        'x_max_km': float(x_max),
        'z_min_km': 0.0,
        'z_max_km': float(z_max),
        'v0_km_s': float(DEFAULT_V0),
        'v1_km_s': float(DEFAULT_V1),
        'n_sources': int(n_src),
        'n_receivers': int(n_rec),
        'noise_std_s': float(noise),
    }
    with open(output_dir / 'meta_data.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Data saved to {output_dir}/")
    return {
        'velocity_true': v_true,
        'traveltime_obs': T_obs,
        'sources': sources,
        'receivers': receivers,
        'meta': meta,
    }


if __name__ == '__main__':
    generate_data()
