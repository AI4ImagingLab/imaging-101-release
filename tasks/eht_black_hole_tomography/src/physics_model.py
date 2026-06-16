"""
Physics model for BH-NeRF: gravitational lensing, Keplerian dynamics, and volume rendering.

Ported from:
- bhnerf/utils.py (rotation_matrix)
- bhnerf/emission.py (velocity_warp_coords, fill_unsupervised_emission, interpolate_coords)
- bhnerf/kgeo.py (radiative_trasfer, doppler_factor, azimuthal_velocity_vector)

Reference: Levis et al. "Gravitationally Lensed Black Hole Emission Tomography" (CVPR 2022)
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotation matrix (Rodrigues formula)
# Adapted from bhnerf/utils.py rotation_matrix()
# ---------------------------------------------------------------------------

def rotation_matrix(axis, angle):
    """
    Rotation matrix via Euler-Rodrigues formula.

    Parameters
    ----------
    axis : array-like, shape (3,)
        Unit rotation axis.
    angle : np.ndarray or float
        Rotation angle(s) in radians. Scalar or arbitrary shape (*batch).

    Returns
    -------
    R : np.ndarray, shape (3, 3, *batch) or (3, 3)
        Rotation matrix/matrices.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.sqrt(np.dot(axis, axis))
    angle = np.asarray(angle, dtype=np.float64)

    a = np.cos(angle / 2.0)
    b, c, d = [-ax * np.sin(angle / 2.0) for ax in axis]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
    ])


def rotation_matrix_torch(axis, angle):
    """
    Torch version of rotation_matrix for differentiable computation.

    Parameters
    ----------
    axis : torch.Tensor, shape (3,)
    angle : torch.Tensor, shape (*batch)

    Returns
    -------
    R : torch.Tensor, shape (3, 3, *batch)
    """
    axis = axis / torch.norm(axis)
    a = torch.cos(angle / 2.0)
    b, c, d = [-ax * torch.sin(angle / 2.0) for ax in axis]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return torch.stack([
        torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)]),
        torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)]),
        torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]),
    ])


# ---------------------------------------------------------------------------
# Keplerian angular velocity
# ---------------------------------------------------------------------------

def keplerian_omega(r, spin=0.0, M=1.0):
    """
    Keplerian angular velocity in Boyer-Lindquist coordinates.

    Omega = sqrt(M) / (r^{3/2} + a * sqrt(M))

    Parameters
    ----------
    r : array-like
        Radial coordinate in units of GM/c^2.
    spin : float
        Dimensionless spin parameter (0 for Schwarzschild).
    M : float
        Black hole mass in geometric units (default 1).

    Returns
    -------
    Omega : same type as r
    """
    sqrt_M = np.sqrt(M) if isinstance(r, np.ndarray) else (M ** 0.5)
    if isinstance(r, torch.Tensor):
        return torch.sqrt(torch.tensor(M, dtype=r.dtype, device=r.device)) / (r ** 1.5 + spin * torch.sqrt(torch.tensor(M, dtype=r.dtype, device=r.device)))
    return np.sqrt(M) / (np.asarray(r, dtype=np.float64) ** 1.5 + spin * np.sqrt(M))


# ---------------------------------------------------------------------------
# Velocity warp coordinates (Keplerian orbital shearing)
# Adapted from bhnerf/emission.py velocity_warp_coords()
# ---------------------------------------------------------------------------

def velocity_warp_coords(coords, Omega, t_frame, t_start_obs, t_geo,
                         t_injection, rot_axis=None, GM_c3=1.0):
    """
    Apply Keplerian velocity warp to coordinates.

    Computes the inverse rotation that maps observation-time coordinates
    back to the emission injection time.

    Parameters
    ----------
    coords : torch.Tensor, shape (3, *spatial)
        Cartesian coordinates [x, y, z] along ray paths.
    Omega : torch.Tensor, shape (*spatial)
        Angular velocity at each coordinate point.
    t_frame : float
        Observation time for this frame.
    t_start_obs : float
        Start time of observations.
    t_geo : torch.Tensor, shape (*spatial)
        Coordinate time along each geodesic (slow light).
    t_injection : float
        Time of hotspot injection in M units.
    rot_axis : torch.Tensor, shape (3,), optional
        Rotation axis. Default [0, 0, 1].
    GM_c3 : float
        Unit conversion factor (1.0 if times already in M units).

    Returns
    -------
    warped_coords : torch.Tensor, shape (*spatial, 3)
    """
    if rot_axis is None:
        rot_axis = torch.tensor([0.0, 0.0, 1.0], dtype=coords.dtype,
                                device=coords.device)

    # Time in M units
    t_M = (t_frame - t_start_obs) / GM_c3 + t_geo - t_injection

    # Rotation angle at each point
    theta_rot = t_M * Omega

    # Mask pre-injection times with NaN
    theta_rot = torch.where(t_M < 0.0, torch.full_like(theta_rot, float('nan')),
                            theta_rot)

    # Build inverse rotation matrix: R(axis, -theta)
    inv_rot = rotation_matrix_torch(rot_axis, -theta_rot)  # (3, 3, *spatial)

    # Apply rotation: warped = R @ coords
    # coords: (3, *spatial) -> need einsum
    warped = torch.einsum('ij...,j...->...i', inv_rot, coords)
    return warped


# ---------------------------------------------------------------------------
# Fill unsupervised region
# Adapted from bhnerf/emission.py fill_unsupervised_emission()
# ---------------------------------------------------------------------------

def fill_unsupervised(emission, coords, rmin=0.0, rmax=float('inf'),
                      z_width=2.0):
    """
    Zero emission outside the valid recovery region.

    Parameters
    ----------
    emission : torch.Tensor, shape (*spatial)
    coords : torch.Tensor, shape (3, *spatial) or (*spatial, 3)
        If last dim is 3, interpreted as (..., [x, y, z]).
    rmin, rmax : float
        Radial bounds in M units.
    z_width : float
        Maximum |z| in M units.

    Returns
    -------
    emission : torch.Tensor, same shape
    """
    if coords.shape[-1] == 3:
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    else:
        x, y, z = coords[0], coords[1], coords[2]

    r_sq = x ** 2 + y ** 2 + z ** 2
    mask = (r_sq >= rmin ** 2) & (r_sq <= rmax ** 2) & (torch.abs(z) <= z_width)
    return emission * mask.float()


# ---------------------------------------------------------------------------
# Trilinear interpolation
# ---------------------------------------------------------------------------

def trilinear_interpolate(volume, coords, fov_min, fov_max):
    """
    Trilinear interpolation of a 3D volume at arbitrary coordinates.

    Uses torch.nn.functional.grid_sample internally.

    Parameters
    ----------
    volume : torch.Tensor, shape (D, H, W)
        3D emission volume.
    coords : torch.Tensor, shape (*, 3)
        Query coordinates in world space, each row [x, y, z].
    fov_min, fov_max : float
        World-coordinate bounds of the volume grid.

    Returns
    -------
    values : torch.Tensor, shape (*)
    """
    original_shape = coords.shape[:-1]
    coords_flat = coords.reshape(-1, 3)

    # Map world coords to [-1, 1] for grid_sample
    grid = 2.0 * (coords_flat - fov_min) / (fov_max - fov_min) - 1.0

    # grid_sample expects (N, D_out, H_out, W_out, 3) in (x, y, z) = (W, H, D) order
    # Our coords are (x, y, z) which maps to (D, H, W) indexing='ij'
    # grid_sample uses (x->W, y->H, z->D) convention, so we reverse
    grid = grid.flip(-1)  # (z, y, x) for grid_sample's (x, y, z)=(W, H, D)

    # Reshape for grid_sample: (1, 1, 1, N, 3)
    grid = grid.reshape(1, 1, 1, -1, 3)

    vol = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    sampled = F.grid_sample(vol, grid, mode='bilinear', padding_mode='zeros',
                            align_corners=True)
    # sampled: (1, 1, 1, 1, N)
    values = sampled.reshape(original_shape)
    return values


# ---------------------------------------------------------------------------
# Volume rendering (radiative transfer)
# Adapted from bhnerf/kgeo.py radiative_trasfer()
# ---------------------------------------------------------------------------

def volume_render(emission, g, dtau, Sigma):
    """
    Integrate emission along rays.

    I = sum(g^2 * emission * dtau * Sigma, axis=-1)

    Parameters
    ----------
    emission : torch.Tensor, shape (*spatial, ngeo)
    g : torch.Tensor, shape (*spatial, ngeo)
        Doppler factor.
    dtau : torch.Tensor, shape (*spatial, ngeo) or broadcastable
        Mino time differential.
    Sigma : torch.Tensor, shape (*spatial, ngeo) or broadcastable
        Metric factor.

    Returns
    -------
    image : torch.Tensor, shape (*spatial)
    """
    return (g ** 2 * emission * dtau * Sigma).sum(dim=-1)


# ---------------------------------------------------------------------------
# DFT measurement matrix
# ---------------------------------------------------------------------------

def dft_matrix(uv_coords, fov_rad, npix):
    """
    Build discrete Fourier transform matrix for VLBI measurement.

    A[k, j] = exp(-2*pi*i*(u_k*l_j + v_k*m_j))

    Parameters
    ----------
    uv_coords : np.ndarray, shape (M, 2)
        Baseline coordinates in wavelengths.
    fov_rad : float
        Field of view in radians.
    npix : int
        Number of image pixels per side.

    Returns
    -------
    A : np.ndarray, shape (M, npix*npix), complex
    """
    pixel_size = fov_rad / npix
    coords_1d = (np.arange(npix) - npix / 2.0) * pixel_size
    ll, mm = np.meshgrid(coords_1d, coords_1d, indexing='ij')
    ll_flat = ll.ravel()
    mm_flat = mm.ravel()

    u = uv_coords[:, 0]
    v = uv_coords[:, 1]

    phase = -2.0 * np.pi * (u[:, None] * ll_flat[None, :] +
                             v[:, None] * mm_flat[None, :])
    A = np.exp(1j * phase)
    return A


# ---------------------------------------------------------------------------
# Forward model class
# ---------------------------------------------------------------------------

class ForwardModel:
    """
    Complete forward model: emission_3d -> warp -> interpolate -> volume_render -> images.

    Parameters
    ----------
    ray_coords : torch.Tensor, shape (3, num_alpha, num_beta, ngeo)
    Omega : torch.Tensor, shape (num_alpha, num_beta, ngeo)
    g_doppler : torch.Tensor, shape (num_alpha, num_beta, ngeo)
    dtau : torch.Tensor, shape (num_alpha, num_beta, ngeo)
    Sigma : torch.Tensor, shape (num_alpha, num_beta, ngeo)
    t_geo : torch.Tensor, shape (num_alpha, num_beta, ngeo)
    fov_M : float
    """

    def __init__(self, ray_coords, Omega, g_doppler, dtau, Sigma, t_geo, fov_M):
        self.ray_coords = ray_coords
        self.Omega = Omega
        self.g_doppler = g_doppler
        self.dtau = dtau
        self.Sigma = Sigma
        self.t_geo = t_geo
        self.fov_M = fov_M

    def render_frame(self, emission_3d, t_frame, t_start_obs, t_injection,
                     rot_axis=None):
        """
        Render a single image frame from a 3D emission volume.

        Parameters
        ----------
        emission_3d : torch.Tensor, shape (D, H, W)
        t_frame : float
        t_start_obs : float
        t_injection : float
        rot_axis : torch.Tensor, shape (3,), optional

        Returns
        -------
        image : torch.Tensor, shape (num_alpha, num_beta)
        """
        warped = velocity_warp_coords(
            self.ray_coords, self.Omega, t_frame, t_start_obs,
            self.t_geo, t_injection, rot_axis=rot_axis
        )

        # Interpolate emission at warped coordinates
        fov_min = -self.fov_M / 2.0
        fov_max = self.fov_M / 2.0

        # Handle NaN (pre-injection) by replacing with zeros
        valid_mask = torch.isfinite(warped).all(dim=-1)
        warped_safe = torch.where(valid_mask.unsqueeze(-1), warped,
                                  torch.zeros_like(warped))

        emission_vals = trilinear_interpolate(
            emission_3d, warped_safe, fov_min, fov_max
        )
        emission_vals = emission_vals * valid_mask.float()

        # Volume render
        image = volume_render(emission_vals, self.g_doppler, self.dtau,
                              self.Sigma)
        return image

    def render_movie(self, emission_3d, t_frames, t_start_obs, t_injection,
                     rot_axis=None):
        """
        Render a full movie from a 3D emission volume.

        Returns
        -------
        images : torch.Tensor, shape (n_frames, num_alpha, num_beta)
        """
        images = []
        for t in t_frames:
            images.append(self.render_frame(emission_3d, float(t),
                                            t_start_obs, t_injection,
                                            rot_axis))
        return torch.stack(images)
