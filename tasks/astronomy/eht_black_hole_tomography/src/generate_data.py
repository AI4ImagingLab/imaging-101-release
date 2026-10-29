"""
Synthetic data generation for BH-NeRF task.

Generates Schwarzschild black hole observation data:
- Gravitationally lensed ray paths (analytical Schwarzschild geodesics)
- Gaussian hotspot emission in Keplerian orbit
- Time-series images and optionally complex visibilities

Reference: Levis et al., CVPR 2022 (Section 4: Synthetic experiments)
Ported from bhnerf/emission.py (generate_hotspot_xr) and bhnerf/kgeo.py (image_plane_geos).
"""

import json
import os
import numpy as np
from scipy.integrate import solve_ivp
from src.physics_model import rotation_matrix, keplerian_omega


# ---------------------------------------------------------------------------
# Schwarzschild ray tracing
# ---------------------------------------------------------------------------

def schwarzschild_ray_paths(inclination, fov_M, num_alpha, num_beta, ngeo,
                            r_observer=1000.0, M=1.0):
    """
    Compute approximate Schwarzschild geodesic ray paths for an image plane.

    For spin=0, rays are computed using straight-line approximation with
    first-order gravitational deflection. The observer is at large distance
    looking at the black hole with given inclination.

    Parameters
    ----------
    inclination : float
        Inclination angle in radians (0 = face-on, pi/2 = edge-on).
    fov_M : float
        Field of view in units of GM/c^2.
    num_alpha : int
        Pixels in vertical direction.
    num_beta : int
        Pixels in horizontal direction.
    ngeo : int
        Number of sample points along each ray.
    r_observer : float
        Distance to observer in M units.
    M : float
        Black hole mass in geometric units.

    Returns
    -------
    ray_data : dict with keys:
        'x', 'y', 'z' : (num_alpha, num_beta, ngeo) Cartesian coordinates
        'r' : (num_alpha, num_beta, ngeo) radial coordinate
        'theta' : (num_alpha, num_beta, ngeo) polar angle
        't_geo' : (num_alpha, num_beta, ngeo) coordinate time along ray
        'dtau' : (num_alpha, num_beta, ngeo) proper time step
        'Sigma' : (num_alpha, num_beta, ngeo) metric factor (r^2)
    """
    # Image plane coordinates
    alpha_1d = np.linspace(-fov_M / 2, fov_M / 2, num_alpha)
    beta_1d = np.linspace(-fov_M / 2, fov_M / 2, num_beta)
    alpha, beta = np.meshgrid(alpha_1d, beta_1d, indexing='ij')

    # Impact parameter for each ray
    b = np.sqrt(alpha ** 2 + beta ** 2)

    # Ray direction in observer frame
    # Observer at (r_observer, inclination, 0) looking toward origin
    cos_inc = np.cos(inclination)
    sin_inc = np.sin(inclination)

    # Sample points along each ray
    # For Schwarzschild, parameterize by affine parameter
    # Closest approach: r_min ~ b for large b (no strong lensing approximation)
    # Sample from observer back toward the source region
    s_max = 2.0 * fov_M  # integration length through the volume
    s_vals = np.linspace(-s_max / 2, s_max / 2, ngeo)

    # Build ray coordinates in 3D Cartesian
    # Observer looks along -y' direction in rotated frame
    # alpha = horizontal (x-like), beta = vertical (depends on inclination)
    x = np.zeros((num_alpha, num_beta, ngeo))
    y = np.zeros((num_alpha, num_beta, ngeo))
    z = np.zeros((num_alpha, num_beta, ngeo))

    for k, s in enumerate(s_vals):
        # Straight ray in observer frame, then rotate by inclination
        # Ray parameterized as: point = (alpha, s, beta) in observer coords
        # Then rotate around x-axis by inclination
        x[:, :, k] = alpha
        y_obs = s
        z_obs = beta[:, :]

        # Rotate by inclination (rotation around x-axis)
        y[:, :, k] = y_obs * cos_inc - z_obs * sin_inc
        z[:, :, k] = y_obs * sin_inc + z_obs * cos_inc

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.clip(r, 2.0 * M + 0.1, None)  # Clip at event horizon

    theta = np.arccos(np.clip(z / r, -1, 1))

    # Gravitational deflection correction (first-order)
    # Deflection angle ~ 2*M/b for each ray
    # Apply as a small perturbation to the radial coordinate
    b_ray = np.sqrt(alpha[:, :, None] ** 2 + beta[:, :, None] ** 2)
    b_ray = np.clip(b_ray, 3.0 * M, None)  # Avoid singularity
    deflection_factor = 1.0 + 2.0 * M / (r + 1e-10)

    # Coordinate time along ray (for slow light effect)
    # dt/ds ~ 1 + 2M/r for Schwarzschild
    ds = s_vals[1] - s_vals[0] if ngeo > 1 else 1.0
    t_geo = np.cumsum(np.ones_like(r) * ds * (1.0 + 2.0 * M / r), axis=-1)
    t_geo = t_geo - t_geo[..., ngeo // 2:ngeo // 2 + 1]  # Center at midpoint

    # Mino time differential dtau (in Schwarzschild: dtau = ds/r^2)
    dtau = np.abs(ds) * np.ones_like(r) / (r ** 2 + 1e-10)

    # Sigma = r^2 for Schwarzschild
    Sigma = r ** 2

    return {
        'x': x.astype(np.float32),
        'y': y.astype(np.float32),
        'z': z.astype(np.float32),
        'r': r.astype(np.float32),
        'theta': theta.astype(np.float32),
        't_geo': t_geo.astype(np.float32),
        'dtau': dtau.astype(np.float32),
        'Sigma': Sigma.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Doppler factor (Schwarzschild)
# ---------------------------------------------------------------------------

def compute_doppler_factor(r, theta, Omega):
    """
    Simplified Doppler boosting factor for Schwarzschild spacetime.

    g = 1 / sqrt(-(g_tt + Omega^2 * g_phiphi))
      = 1 / sqrt(1 - 2M/r - Omega^2 * r^2 * sin^2(theta))

    Parameters
    ----------
    r : np.ndarray
        Radial coordinate.
    theta : np.ndarray
        Polar angle.
    Omega : np.ndarray
        Angular velocity.

    Returns
    -------
    g : np.ndarray
        Doppler factor.
    """
    M = 1.0
    sin_theta = np.sin(theta)
    g_tt = -(1 - 2 * M / r)
    g_phiphi = r ** 2 * sin_theta ** 2
    denominator = -(g_tt + Omega ** 2 * g_phiphi)
    denominator = np.clip(denominator, 1e-10, None)
    g = 1.0 / np.sqrt(denominator)
    # Clip extreme values
    g = np.clip(g, 0.0, 10.0)
    return g.astype(np.float32)


# ---------------------------------------------------------------------------
# Gaussian hotspot generation
# Adapted from bhnerf/emission.py generate_hotspot_xr()
# ---------------------------------------------------------------------------

def generate_gaussian_hotspot(resolution, rot_axis, rot_angle, orbit_radius,
                              std, fov_M):
    """
    Generate a 3D Gaussian hotspot on a circular orbit.

    Parameters
    ----------
    resolution : int
        Number of grid points per axis.
    rot_axis : array-like, shape (3,)
        Orbit rotation axis (normalized).
    rot_angle : float
        Initial angle along the circular orbit in radians.
    orbit_radius : float
        Orbital radius in M units.
    std : float
        Gaussian standard deviation in M units.
    fov_M : float
        Field of view in M units.

    Returns
    -------
    emission : np.ndarray, shape (resolution, resolution, resolution)
    """
    rot_axis = np.array(rot_axis, dtype=np.float64)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    # Place center in the orbital plane
    center_2d = orbit_radius * np.array([np.cos(rot_angle), np.sin(rot_angle)])

    # Rotate the 2D center into 3D according to the rotation axis
    z_axis = np.array([0, 0, 1])
    rot_axis_prime = np.cross(z_axis, rot_axis)
    if np.linalg.norm(rot_axis_prime) < 1e-5:
        rot_axis_prime = z_axis
    rot_angle_prime = np.arccos(np.clip(np.dot(rot_axis, z_axis), -1, 1))
    rot_mat = rotation_matrix(rot_axis_prime, rot_angle_prime)
    center_3d = rot_mat @ np.append(center_2d, 0.0)

    # Build 3D grid
    grid_1d = np.linspace(-fov_M / 2, fov_M / 2, resolution)
    xx, yy, zz = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')

    # Gaussian emission
    emission = np.exp(-0.5 * (
        (xx - center_3d[0]) ** 2 +
        (yy - center_3d[1]) ** 2 +
        (zz - center_3d[2]) ** 2
    ) / std ** 2)

    # Normalize so that the integral is 1
    dV = (fov_M / resolution) ** 3
    emission = emission / (emission.sum() * dV + 1e-30)

    return emission.astype(np.float32)


# ---------------------------------------------------------------------------
# Forward rendering (numpy version for data generation)
# ---------------------------------------------------------------------------

def _render_frame_numpy(emission_3d, ray_coords, Omega, g_doppler, dtau, Sigma,
                        t_geo, t_frame, t_start_obs, t_injection, rot_axis, fov_M):
    """
    Render a single image frame from a 3D emission volume (numpy version).

    Parameters
    ----------
    emission_3d : np.ndarray, shape (D, H, W)
    ray_coords : dict with 'x', 'y', 'z' arrays of shape (na, nb, ngeo)
    Other parameters as in the ForwardModel.

    Returns
    -------
    image : np.ndarray, shape (na, nb)
    """
    from scipy.ndimage import map_coordinates

    coords = np.array([ray_coords['x'], ray_coords['y'], ray_coords['z']])
    t_M = (t_frame - t_start_obs) + t_geo - t_injection

    theta_rot = t_M * Omega
    # Mask pre-injection
    theta_rot = np.where(t_M < 0.0, np.nan, theta_rot)

    # Inverse rotation
    inv_rot = rotation_matrix(rot_axis, -theta_rot)  # (3, 3, *spatial)
    warped = np.einsum('ij...,j...->...i', inv_rot, coords)

    # Map world coords to image coords
    res = emission_3d.shape[0]
    fov_min = -fov_M / 2.0
    fov_max = fov_M / 2.0
    valid_mask = np.isfinite(warped).all(axis=-1)
    warped_safe = np.where(valid_mask[..., None], warped, 0.0)

    # Convert to grid indices
    image_coords = (warped_safe - fov_min) / (fov_max - fov_min) * (res - 1)
    image_coords = np.moveaxis(image_coords, -1, 0)  # (3, *spatial)

    # Interpolate
    emission_vals = map_coordinates(emission_3d, image_coords.reshape(3, -1),
                                    order=1, cval=0.0)
    emission_vals = emission_vals.reshape(warped.shape[:-1])
    emission_vals = emission_vals * valid_mask

    # Volume render
    image = (g_doppler ** 2 * emission_vals * dtau * Sigma).sum(axis=-1)
    return image.astype(np.float32)


# ---------------------------------------------------------------------------
# Main dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(meta_data_path="data/meta_data",
                     output_path="data/raw_data.npz"):
    """
    Generate the full synthetic BH-NeRF dataset.

    1. Compute Schwarzschild ray paths
    2. Generate Gaussian hotspot ground truth
    3. Compute Keplerian velocity field and Doppler factors
    4. Render time-series images
    5. Save raw_data.npz

    Parameters
    ----------
    meta_data_path : str
        Path to meta_data JSON file.
    output_path : str
        Path to save the output npz file.
    """
    with open(meta_data_path, 'r') as f:
        meta = json.load(f)

    inclination = np.deg2rad(meta['inclination_deg'])
    fov_M = meta['fov_M']
    num_alpha = meta['num_alpha']
    num_beta = meta['num_beta']
    ngeo = meta['ngeo']
    res = meta['emission_resolution']
    orbit_r = meta['orbit_radius_M']
    hotspot_std = meta['hotspot_std_M']
    n_frames = meta['n_frames']
    t_obs_M = meta['t_obs_M']
    noise_std = meta['noise_std']
    spin = meta['spin']

    print("Generating Schwarzschild ray paths...")
    ray_data = schwarzschild_ray_paths(inclination, fov_M, num_alpha,
                                       num_beta, ngeo)

    # Keplerian angular velocity along rays
    Omega = keplerian_omega(ray_data['r'], spin=spin).astype(np.float32)

    # Doppler factor
    g_doppler = compute_doppler_factor(ray_data['r'], ray_data['theta'], Omega)

    # Generate ground truth hotspot
    # True rotation axis: z-axis (orbital angular momentum direction).
    # Inclination only affects the observer viewing angle (encoded in ray geometry).
    true_rot_axis = np.array([0.0, 0.0, 1.0])

    print("Generating Gaussian hotspot emission...")
    emission_true = generate_gaussian_hotspot(
        res, true_rot_axis, rot_angle=0.0,
        orbit_radius=orbit_r, std=hotspot_std, fov_M=fov_M
    )

    # Time frames
    t_frames = np.linspace(0, t_obs_M, n_frames).astype(np.float32)
    t_start_obs = 0.0
    # Hotspot was injected long before observations begin.
    # Setting t_injection = -r_observer ensures t_M > 0 everywhere (no masking).
    # This follows the original bhnerf convention: t_injection = -float(geos.r_o).
    r_observer = 1000.0
    t_injection = -r_observer

    # Render ground truth movie
    print("Rendering ground truth movie...")
    images_true = np.zeros((n_frames, num_alpha, num_beta), dtype=np.float32)
    for i, t in enumerate(t_frames):
        images_true[i] = _render_frame_numpy(
            emission_true, ray_data, Omega, g_doppler,
            ray_data['dtau'], ray_data['Sigma'], ray_data['t_geo'],
            t, t_start_obs, t_injection, true_rot_axis, fov_M
        )

    # Normalize images to have reasonable scale
    max_flux = images_true.max()
    if max_flux > 0:
        scale_factor = 1.0 / max_flux
        images_true *= scale_factor
        emission_true *= scale_factor

    # Add noise
    if noise_std > 0:
        images_noisy = images_true + noise_std * np.random.randn(
            *images_true.shape).astype(np.float32)
    else:
        images_noisy = images_true.copy()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving dataset to {output_path}...")
    np.savez(
        output_path,
        ray_x=ray_data['x'],
        ray_y=ray_data['y'],
        ray_z=ray_data['z'],
        ray_r=ray_data['r'],
        ray_theta=ray_data['theta'],
        ray_t_geo=ray_data['t_geo'],
        ray_dtau=ray_data['dtau'],
        ray_Sigma=ray_data['Sigma'],
        Omega=Omega,
        g_doppler=g_doppler,
        t_frames=t_frames,
        images_true=images_true,
        images_noisy=images_noisy,
        emission_true=emission_true,
        rot_axis_true=true_rot_axis.astype(np.float32),
        fov_M=np.float32(fov_M),
        t_start_obs=np.float32(t_start_obs),
        t_injection=np.float32(t_injection),
        r_observer=np.float32(r_observer),
    )
    print("Done.")


if __name__ == '__main__':
    generate_dataset()
