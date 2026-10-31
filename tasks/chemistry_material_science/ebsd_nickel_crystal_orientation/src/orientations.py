"""MIT2025 ZacharyVarley. Selected explicit numerical kernels extracted from
pcadi commit88f676e6f44812900b3c873692aa1a1c9e48b746.
Only the bound cubic-Ni mathematics is retained; no domain-library import,
quantization framework, IO package, or generic vendor class is included.
"""
import math
import torch
from torch import Tensor
from typing import Tuple,Optional
def qu_std(qu: Tensor) -> Tensor:
    """
    Standardize unit quaternion to have non-negative real part.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(qu[..., 0:1] >= 0, qu, -qu)

def qu_norm(qu: Tensor) -> Tensor:
    """
    Normalize quaternions to unit norm.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Tensor of normalized quaternions.
    """
    return qu / torch.norm(qu, dim=-1, keepdim=True)

def qu_prod_raw(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = (a[..., 0], a[..., 1], a[..., 2], a[..., 3])
    bw, bx, by, bz = (b[..., 0], b[..., 1], b[..., 2], b[..., 3])
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def qu_prod(a: Tensor, b: Tensor) -> Tensor:
    """
    Quaternion multiplication, then make real part non-negative.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b Tensor shape (..., 4) of the quaternion product.

    """
    ab = qu_prod_raw(a, b)
    return qu_std(ab)

def qu_prod_pos_real(a: Tensor, b: Tensor) -> Tensor:
    """
    Return only the magnitude of the real part of the quaternion product.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b Tensor shape (..., ) of quaternion product real part magnitudes.
    """
    aw, ax, ay, az = (a[..., 0], a[..., 1], a[..., 2], a[..., 3])
    bw, bx, by, bz = (b[..., 0], b[..., 1], b[..., 2], b[..., 3])
    ow = aw * bw - ax * bx - ay * by - az * bz
    return ow.abs()

def qu_conj(qu: Tensor) -> Tensor:
    """
    Get the unit quaternions for the inverse action.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    scaling = torch.tensor([1, -1, -1, -1], device=qu.device, dtype=qu.dtype)
    return qu * scaling

def qu_apply(qu: Tensor, point: Tensor) -> Tensor:
    """
    Rotate 3D points by unit quaternions.

    Args:
        qu: shape (..., 4) of quaternions in the form (w, x, y, z)
        point: shape (..., 3) of 3D points.

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    aw, ax, ay, az = (qu[..., 0], qu[..., 1], qu[..., 2], qu[..., 3])
    bx, by, bz = (point[..., 0], point[..., 1], point[..., 2])
    iw = aw - ax * bx - ay * by - az * bz
    ix = aw * bx + ax + ay * bz - az * by
    iy = aw * by - ax * bz + ay + az * bx
    iz = aw * bz + ax * by - ay * bx + az
    ox = -iw * ax + ix * aw - iy * az + iz * ay
    oy = -iw * ay + ix * az + iy * aw - iz * ax
    oz = -iw * az - ix * ay + iy * ax + iz * aw
    return torch.stack((ox, oy, oz), -1)

def qu_norm_std(qu: Tensor) -> Tensor:
    """
    Normalize a quaternion to unit norm and make real part non-negative.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Tensor of normalized and standardized quaternions.
    """
    return qu_std(qu_norm(qu))

def cu2ho(cu: torch.Tensor) -> torch.Tensor:
    """
    Converts cubochoric vector representation to homochoric vector representation.

    Args:
        cu: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Homochoric vectors as tensor of shape (..., 3).
    """
    indices = torch.argsort(torch.abs(cu), dim=-1, descending=False)
    sorted = torch.gather(cu, -1, indices)
    s, m, b = sorted.unbind(dim=-1)
    trig_arg_xy = s * torch.pi / (12.0 * m)
    trig_arg_xy[torch.isnan(trig_arg_xy)] = 0
    factor_xy = 2 ** (1 / 12) * 3 ** (1 / 3) * m * torch.sqrt((4 * b ** 2 * (torch.cos(trig_arg_xy) - 2 ** 0.5) + 2 ** 0.5 * m ** 2 * (-2 * 2 ** 0.5 * torch.cos(trig_arg_xy) + 3)) / (torch.cos(trig_arg_xy) - 2 ** 0.5)) / (torch.pi ** (1 / 3) * b * torch.sqrt(-torch.cos(trig_arg_xy) + 2 ** 0.5))
    x_s3 = factor_xy * torch.sin(trig_arg_xy)
    y_s3 = factor_xy * 2 ** (-0.5) * (2 ** 0.5 * torch.cos(trig_arg_xy) - 1)
    z_s3 = (2 * 6 ** (1 / 3) * b ** 2 * (torch.cos(trig_arg_xy) - 2 ** 0.5) + 2 ** (5 / 6) * 3 ** (1 / 3) * m ** 2 * (-2 * 2 ** 0.5 * torch.cos(trig_arg_xy) + 3)) / (2 * torch.pi ** (1 / 3) * b * (torch.cos(trig_arg_xy) - 2 ** 0.5))
    ho = torch.stack((x_s3, y_s3, z_s3), dim=-1)
    ho = torch.scatter(ho, -1, indices, ho)
    ho[torch.isnan(ho)] = 0
    ho.copysign_(cu)
    return ho

def ho2ax(ho: Tensor, fast: bool=True) -> Tensor:
    """
    Converts a set of homochoric vectors to axis-angle representation.

    Args:
        ho (Tensor): shape (..., 3) homochoric coordinates (x, y, z)
        fast (bool): by default skip Newton iteration for FP64 only

    Returns:
        torch.Tensor: shape (..., 4) axis-angles (x, y, z, angle)


    Notes:

    These are Chebyshev fits on the modified homochoric inverse
    fitted by Zachary Varley on 05/24/2024. The modified homochoric
    inverse ties the square of the homochoric vector back to the
    cosine of the half rotation angle.

    """
    if ho.dtype == torch.float32 or ho.dtype == torch.float16:
        fit_parameters = torch.tensor([1.0000000000000009, -0.49999943403867775, -0.02501516506014902, -0.003812013154855173, -0.0012106188330642162, 0.0004932929599315542, -0.0007008938552645062, 0.0003097977492358908, -7.302347496329884e-05], dtype=ho.dtype, device=ho.device).to(ho.dtype)
    else:
        fit_parameters = torch.tensor([1.0, -0.49999997124013285, -0.025001181866044025, -0.003914420982052104, -0.0008932026810453948, 3.1181024286083695e-05, -0.00043961032788396477, 0.0003965747172750644, -0.0002637994505058693, 9.118535597958716e-05, -1.487586780569253e-05], dtype=ho.dtype, device=ho.device).to(ho.dtype)
    ho_norm_sq = torch.sum(ho ** 2, dim=-1, keepdim=False)
    s = torch.zeros_like(ho_norm_sq)
    for i in range(len(fit_parameters)):
        s += fit_parameters[i] * ho_norm_sq ** i
    if ho.dtype == torch.float64 and (not fast):
        w = 2 * torch.arccos(torch.clamp(s, -1.0, 1.0))
        f_w = (3 / 4 * (w - torch.sin(w))) ** (1 / 3) - torch.sqrt(ho_norm_sq)
        f_p_w = (1 - torch.cos(w)) / (6 ** (2 / 3) * (w - torch.sin(w)) ** (2 / 3))
        update = f_w / f_p_w
        update[torch.isnan(update)] = 0
        w -= update
    else:
        w = 2.0 * torch.arccos(torch.clamp(s, -1.0, 1.0))
    ax = torch.concat([ho * torch.rsqrt(ho_norm_sq).unsqueeze(-1), w.unsqueeze(-1)], dim=-1)
    rot_is_identity = torch.abs(ho_norm_sq) < 1e-06
    ax[rot_is_identity] = 0
    ax[rot_is_identity, ..., 2] = 1.0
    return ax

def ax2qu(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to quaternion representation.

    Args:
        ax (Tensor): shape (..., 4) axis-angle in the format (x, y, z, angle).

    Returns:
        torch.Tensor: shape (..., 4) quaternions in the format (w, x, y, z).
    """
    qu = torch.empty_like(ax)
    cos_half_ang = torch.cos(ax[..., 3] / 2.0)
    sin_half_ang = torch.sin(ax[..., 3:4] / 2.0)
    qu[..., 0] = cos_half_ang
    qu[..., 1:] = ax[..., :3] * sin_half_ang
    return qu

def cu2qu(cubochoric_vectors: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to quaternions.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return ax2qu(ho2ax(cu2ho(cubochoric_vectors)))

def so3_cubochoric_grid(edge_length: int, device: torch.device, dtype: torch.dtype=torch.float32) -> Tensor:
    """
    Generate a 3D grid sampling in cubochoric coordinates. Orientations
    are returned as unit quaternions with positive scalar part (w, x, y, z)

    Args:
        :edge_length (int): the number of points along each axis of the cube
        :device (torch.device): the device to use

    Returns:
        torch.Tensor: the 3D grid sampling in cubochoric coordinates (n, 3)

    """
    cu = torch.linspace(-0.5 * torch.pi ** (2.0 / 3.0), 0.5 * torch.pi ** (2.0 / 3.0), edge_length + 1, dtype=dtype, device=device)
    cu = cu[:-1]
    cu = torch.stack(torch.meshgrid(cu, cu, cu, indexing='ij'), dim=-1).reshape(-1, 3)
    qu = cu2qu(cu)
    qu = qu_std(qu)
    return qu

def get_local_orientation_grid(semi_edge_in_degrees: float, kernel_radius_in_steps: int, axial_grid_dimension: int=3) -> Tensor:
    """
    Get the local orientation grid using cubochoric coordinates. The grid will
    be a cube with 2*semi_edge_in_degrees side length and (2*kernel_radius +
    1)^3 grid points.

    The cubochoric box with the identity orientation centered at the origin
    extends from -0.5 * (pi)^(2/3) to 0.5 * (pi)^(2/3) along each axis. The
    points on the outside surface are 180 degree rotations.

    A convenient equal volume subgrid around the origin can be mapped to
    quaternions and used to explore the space of orientations around the initial
    guess via rotational composition.

    Args:
        :semi_edge_in_degrees (float): The semi-edge of the grid in degrees.
        :kernel_radius_in_steps (int): Divisions per side of the kernel.
        :axial_grid_dimension (int): Defaults to 3 (complete outer product). 2
        excluded cube corners. 1 only has the axial points.

    Returns:
        cu (Tensor): The local orientation grid in cubochoric coordinates.

    """
    semi_edge_cubochoric = semi_edge_in_degrees * torch.pi ** (2 / 3) / 360.0
    cu = torch.linspace(-semi_edge_cubochoric, semi_edge_cubochoric, 2 * kernel_radius_in_steps + 1)
    cu = torch.stack(torch.meshgrid(cu, cu, cu, indexing='ij'), dim=-1).reshape(-1, 3)
    if axial_grid_dimension == 3:
        pass
    elif axial_grid_dimension == 2:
        mask = torch.any(cu == 0, dim=-1)
        cu = cu[mask]
    elif axial_grid_dimension == 1:
        mask = torch.sum(cu == 0, dim=-1) >= 2
        cu = cu[mask]
    else:
        raise ValueError('axial_grid_dimension must be 1, 2, or 3.')
    return cu

def cubic_elements():
    R2=1.0/(2.0**0.5)
    LAUE_O = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [R2, 0.0, 0.0, R2], [R2, 0.0, 0.0, -R2], [0.0, R2, R2, 0.0], [0.0, -R2, R2, 0.0], [0.5, 0.5, -0.5, 0.5], [0.5, 0.5, 0.5, -0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, -0.5, -0.5], [0.5, -0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [R2, R2, 0.0, 0.0], [R2, -R2, 0.0, 0.0], [R2, 0.0, R2, 0.0], [R2, 0.0, -R2, 0.0], [0.0, R2, 0.0, R2], [0.0, -R2, 0.0, R2], [0.0, 0.0, R2, R2], [0.0, 0.0, -R2, R2]], dtype=torch.float64)
    return LAUE_O

def sample_cubic_grid(resolution_deg=2.,permute=True):
    """Original equal-volume cubochoric sampling and high-cubic FZ rejection."""
    if resolution_deg<=.03732:raise ValueError('Resolution too small for original empirical grid formula')
    N=int(round(131.97049/(resolution_deg-.03732)));edge=2*N+1
    q=so3_cubochoric_grid(edge,device=torch.device('cpu'))
    xyz=q[...,1:].abs();keep=(xyz.max(-1).values<=q[...,0]*(2**.5-1))&(xyz.sum(-1)<=q[...,0]);q=q[keep]
    if permute:q=q[torch.randperm(len(q))]
    return q
def cubic_reduce(q):
    """Original right-action search proposal remapping, not a forward-equivalence claim."""
    shape=q.shape;g=cubic_elements().to(q.dtype).to(q.device);a=q.reshape(-1,4)
    w=qu_prod_pos_real(a[:,None,:],g);indices=w.argmax(-1)
    return qu_prod(a,g[indices]).reshape(shape)
