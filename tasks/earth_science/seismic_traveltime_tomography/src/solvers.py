"""Adjoint-state traveltime tomography solver.

Implements the ATT inversion loop from Chen et al. (2024) / TomoATT
for the 2D isotropic Cartesian case:

1. Forward step — solve the Eikonal equation from every source.
2. Residual computation — T_syn(x_rec) − T_obs.
3. Sensitivity kernel — accumulate ray-based back-projection kernels:

       K_s(x) = Σ_{n,m}  R_{n,m}  ∫_{ray(n,m)} δ(x − y) dl

   Rays are traced backward from each receiver to the source along the
   direction −∇T_n (steepest descent on the traveltime field).

4. Kernel density normalization (Eq. 18-19 of the paper):

       K_s(x) ← K_s(x) / (K_d(x) + ε)^ζ

   where K_d accumulates total ray coverage independent of residuals.

5. Model update — step-size-controlled gradient descent:

       s_{n+1}(x) = s_n(x) − α · K_s(x)

   The step size α is chosen so that the maximum fractional slowness
   change equals ``step_size`` (e.g. 2%).
"""

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

from .physics_model import (
    solve_eikonal,
    compute_traveltime_at,
    compute_traveltime_gradient,
)
from .preprocessing import compute_residuals, compute_misfit


# --------------------------------------------------------------------------- #
#  Ray tracing                                                                 #
# --------------------------------------------------------------------------- #

def _trace_ray_on_grid(T: np.ndarray,
                       dTdz: np.ndarray, dTdx: np.ndarray,
                       rec_x: float, rec_z: float,
                       src_x: float, src_z: float,
                       dx: float, dz: float,
                       step_km: float) -> tuple:
    """Trace a ray from receiver toward source along −∇T.

    Uses Euler integration with fixed step length.  Terminates when the
    ray comes within ``step_km`` of the source.

    Args:
        T:         (Nz, Nx) traveltime field from source n.
        dTdz, dTdx: Gradient components of T.
        rec_x/z:   Receiver position (km).
        src_x/z:   Source position (km).
        dx, dz:    Grid spacings (km).
        step_km:   Ray step length (km).

    Returns:
        (xs, zs, dls): Lists of x positions, z positions, and step lengths
                       along the ray.
    """
    xs = [rec_x]
    zs = [rec_z]
    dls = []

    pos_x, pos_z = rec_x, rec_z

    max_steps = int(
        (abs(rec_x - src_x) + abs(rec_z - src_z)) / step_km * 4
    ) + 10

    for _ in range(max_steps):
        dist = np.sqrt((pos_x - src_x) ** 2 + (pos_z - src_z) ** 2)
        if dist < step_km:
            xs.append(src_x)
            zs.append(src_z)
            dls.append(dist)
            break

        iz_f = pos_z / dz
        ix_f = pos_x / dx
        coords = [[iz_f], [ix_f]]
        gz = float(map_coordinates(dTdz, coords, order=1, mode='nearest')[0])
        gx = float(map_coordinates(dTdx, coords, order=1, mode='nearest')[0])

        gnorm = np.sqrt(gz ** 2 + gx ** 2)
        if gnorm < 1e-12:
            break

        # Move in direction −∇T / |∇T| (toward source)
        new_x = pos_x - step_km * gx / gnorm
        new_z = pos_z - step_km * gz / gnorm

        # Clamp to domain
        Nz, Nx = T.shape
        new_x = np.clip(new_x, 0.0, (Nx - 1) * dx)
        new_z = np.clip(new_z, 0.0, (Nz - 1) * dz)

        xs.append(new_x)
        zs.append(new_z)
        dls.append(step_km)
        pos_x, pos_z = new_x, new_z

    return xs, zs, dls


# --------------------------------------------------------------------------- #
#  Kernel accumulation                                                         #
# --------------------------------------------------------------------------- #

def compute_sensitivity_kernel(slowness: np.ndarray,
                                dx: float, dz: float,
                                sources: np.ndarray,
                                receivers: np.ndarray,
                                T_obs: np.ndarray,
                                step_km: float = 1.0) -> tuple:
    """Compute sensitivity kernel and kernel density by ray back-projection.

    For each source n, solves the Eikonal equation, computes traveltime
    residuals at all receivers, and traces rays backward from each receiver
    to accumulate the sensitivity kernel.

    Args:
        slowness:   (Nz, Nx) current slowness model (s/km).
        dx, dz:     Grid spacings (km).
        sources:    (N_src, 2) source positions (km).
        receivers:  (N_rec, 2) receiver positions (km).
        T_obs:      (N_src, N_rec) observed traveltimes (s).
        step_km:    Ray integration step length (km).

    Returns:
        kernel:         (Nz, Nx) sensitivity kernel K_s (s/km · km).
        kernel_density: (Nz, Nx) coverage kernel K_d (km).
        T_syn:          (N_src, N_rec) synthetic traveltimes (s).
        misfit:         Scalar misfit χ (s²).
    """
    Nz, Nx = slowness.shape
    kernel         = np.zeros((Nz, Nx), dtype=np.float64)
    kernel_density = np.zeros((Nz, Nx), dtype=np.float64)

    N_src = len(sources)
    N_rec = len(receivers)
    T_syn = np.zeros((N_src, N_rec), dtype=np.float32)

    for n in range(N_src):
        src_x, src_z = sources[n]
        T_n = solve_eikonal(slowness, dx, dz, src_x, src_z)

        for m in range(N_rec):
            T_syn[n, m] = compute_traveltime_at(
                T_n, receivers[m, 0], receivers[m, 1], dx, dz)

        residuals_n = compute_residuals(T_syn[n], T_obs[n])

        # Gradient of traveltime (needed for ray tracing)
        dTdz, dTdx = compute_traveltime_gradient(T_n, dx, dz)

        for m in range(N_rec):
            R_nm = float(residuals_n[m])
            xs, zs, dls = _trace_ray_on_grid(
                T_n, dTdz, dTdx,
                receivers[m, 0], receivers[m, 1],
                src_x, src_z,
                dx, dz, step_km,
            )
            for xp, zp, dl in zip(xs[:-1], zs[:-1], dls):
                ix = int(round(xp / dx))
                iz = int(round(zp / dz))
                ix = np.clip(ix, 0, Nx - 1)
                iz = np.clip(iz, 0, Nz - 1)
                kernel[iz, ix]         += R_nm * dl
                kernel_density[iz, ix] += dl

    misfit = compute_misfit(compute_residuals(T_syn, T_obs))
    return (kernel.astype(np.float32),
            kernel_density.astype(np.float32),
            T_syn,
            misfit)


def kernel_density_normalization(kernel: np.ndarray,
                                 kernel_density: np.ndarray,
                                 zeta: float = 0.5,
                                 epsilon: float = 1e-4) -> np.ndarray:
    """Normalize sensitivity kernel by coverage density (Eq. 18, TomoATT).

    K_s ← K_s / (K_d + ε)^ζ

    This emphasizes sparsely covered regions and accelerates convergence
    where data coverage is low.

    Args:
        kernel:         (Nz, Nx) raw sensitivity kernel K_s.
        kernel_density: (Nz, Nx) coverage kernel K_d.
        zeta:           Normalization exponent (0 = no normalization, 1 = full).
        epsilon:        Small regularization to avoid division by zero.

    Returns:
        kernel_norm: (Nz, Nx) normalized kernel.
    """
    return kernel / (kernel_density + epsilon) ** zeta


def update_slowness(slowness: np.ndarray,
                    kernel_norm: np.ndarray,
                    step_size: float = 0.02) -> np.ndarray:
    """Gradient descent update with step-size control.

    The step size α is set so that the maximum fractional change in
    slowness equals ``step_size`` (e.g., 2%):

        α = step_size * max(s) / max(|K_norm|)

    Args:
        slowness:    (Nz, Nx) current slowness (s/km).
        kernel_norm: (Nz, Nx) normalized sensitivity kernel.
        step_size:   Maximum fractional slowness change per iteration.

    Returns:
        slowness_new: (Nz, Nx) updated slowness (s/km).
    """
    k_max = np.abs(kernel_norm).max()
    if k_max < 1e-30:
        return slowness.copy()

    alpha = step_size * slowness.max() / k_max
    s_new = slowness - alpha * kernel_norm
    # Ensure physical values (v < 15 km/s, v > 1 km/s)
    s_new = np.clip(s_new, 1.0 / 15.0, 1.0 / 1.0)
    return s_new.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Main solver                                                                 #
# --------------------------------------------------------------------------- #

class ATTSolver:
    """Adjoint-state traveltime tomography solver.

    Iteratively updates the slowness model by minimising the traveltime
    misfit χ = Σ_{n,m} (T_syn - T_obs)² / 2 using the sensitivity kernel
    computed by ray back-projection and a step-size-controlled gradient
    descent update.

    Args:
        num_iterations: Number of gradient descent iterations.
        step_size:      Max fractional slowness change per iteration (default 0.02).
        zeta:           Kernel density normalization exponent (default 0.5).
        epsilon:        KD normalization floor (default 1e-4).
        step_km:        Ray tracing step length (km, default 1.0).
        smooth_sigma:   Gaussian smoothing sigma for the kernel (grid cells).
                        Smoothing acts like implicit multiple-grid regularisation,
                        preventing large updates in poorly constrained regions.
    """

    def __init__(self,
                 num_iterations: int = 40,
                 step_size: float = 0.02,
                 step_decay: float = 0.97,
                 zeta: float = 0.5,
                 epsilon: float = 1e-4,
                 step_km: float = 1.0,
                 smooth_sigma: float = 1.5):
        self.num_iterations = num_iterations
        self.step_size      = step_size
        self.step_decay     = step_decay
        self.zeta           = zeta
        self.epsilon        = epsilon
        self.step_km        = step_km
        self.smooth_sigma   = smooth_sigma

    def run(self,
            slowness_init: np.ndarray,
            dx: float, dz: float,
            sources: np.ndarray,
            receivers: np.ndarray,
            T_obs: np.ndarray,
            verbose: bool = True) -> dict:
        """Run the ATT inversion loop.

        Args:
            slowness_init: (Nz, Nx) initial slowness model (s/km).
            dx, dz:        Grid spacings (km).
            sources:       (N_src, 2) source positions (km).
            receivers:     (N_rec, 2) receiver positions (km).
            T_obs:         (N_src, N_rec) observed traveltimes (s).
            verbose:       Print misfit every iteration.

        Returns:
            dict with keys:
                'slowness':      (Nz, Nx) final slowness model.
                'velocity':      (Nz, Nx) final velocity model (km/s).
                'misfit_history': list of scalar misfits per iteration.
                'kernel_final':  final sensitivity kernel.
        """
        slowness = slowness_init.copy().astype(np.float64)
        misfit_history = []
        current_step = self.step_size

        for it in range(self.num_iterations):
            kernel, kernel_density, T_syn, misfit = compute_sensitivity_kernel(
                slowness, dx, dz, sources, receivers, T_obs,
                step_km=self.step_km,
            )
            misfit_history.append(misfit)

            kernel_norm = kernel_density_normalization(
                kernel, kernel_density, self.zeta, self.epsilon)

            if self.smooth_sigma > 0:
                kernel_norm = gaussian_filter(
                    kernel_norm, sigma=self.smooth_sigma).astype(np.float32)

            slowness = update_slowness(
                slowness, kernel_norm, current_step)

            current_step *= self.step_decay

            if verbose:
                print(f"  Iter {it+1:3d}/{self.num_iterations}: misfit = {misfit:.4e} s²")

        # Final forward pass for T_syn
        _, _, T_syn_final, misfit_final = compute_sensitivity_kernel(
            slowness, dx, dz, sources, receivers, T_obs,
            step_km=self.step_km,
        )
        misfit_history.append(misfit_final)

        return {
            'slowness':       slowness.astype(np.float32),
            'velocity':       (1.0 / slowness).astype(np.float32),
            'misfit_history': misfit_history,
            'kernel_final':   kernel_norm,
            'T_syn_final':    T_syn_final,
        }
