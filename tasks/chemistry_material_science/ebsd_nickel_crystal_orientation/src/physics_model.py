"""MIT2025 ZacharyVarley. Selected explicit numerical kernels extracted from
pcadi commit88f676e6f44812900b3c873692aa1a1c9e48b746.
Only the bound cubic-Ni mathematics is retained; no domain-library import,
quantization framework, IO package, or generic vendor class is included.
"""
import math
import torch
from torch import Tensor
from typing import Tuple,Optional
def rosca_lambert_side_by_side(pts: Tensor) -> Tensor:
    """
    Map unit sphere to (-1, 1) X (-1, 1) square via square Rosca-Lambert
    projection. Points with a positive z-coordinate are projected to the left
    side of the square, while points with a negative z-coordinate are projected
    to the right side of the square.

    Args:
        pts: torch tensor of shape (..., 3) containing the points

    Returns:
        torch tensor of shape (..., 2) containing the projected points

    """
    x, y, z = (pts[..., 0], pts[..., 1], pts[..., 2])
    factor = torch.sqrt(torch.clamp(2.0 * (1.0 - torch.abs(z)), min=0.0))
    cond = torch.abs(y) <= torch.abs(x)
    big = torch.where(cond, x, y)
    sml = torch.where(cond, y, x)
    simpler_term = torch.where(big < 0, -1, 1) * factor * (2.0 / 8.0 ** 0.5)
    arctan_term = torch.where(big < 0, -1, 1) * factor * torch.atan2(sml * torch.where(big < 0, -1, 1), torch.abs(big)) * (2.0 * 2.0 ** 0.5 / torch.pi)
    out = torch.stack((simpler_term, arctan_term), dim=-1)
    out = torch.where(cond[..., None], out, out.flip(-1))
    out[..., 0] = out[..., 0] / 2.0 - 0.5 + torch.where(z < 0, 1.0, 0)
    return out

def bruker_geometry_to_SE3(pattern_centers: Tensor, primary_tilt_deg: Tensor, secondary_tilt_deg: Tensor, detector_shape: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
    """
    Convert pattern centers in Bruker coordinates to SE3 transformation matrix.

    Args:
        :pattern_centers (Tensor): The pattern centers.
        :primary_tilt_deg (Tensor): The primary tilt in degrees.
        :secondary_tilt_deg (Tensor): The secondary tilt in degrees.
        :detector_shape (Tuple[int, int]): The detector shape.

    Returns:
        Rotation matrices (..., 3, 3) and translation vectors (..., 3).

    """
    pcx, pcy, pcz = torch.unbind(pattern_centers, dim=-1)
    rows, cols = detector_shape
    rows, cols = (float(rows), float(cols))
    dt_m_sy = torch.deg2rad(primary_tilt_deg)
    sx = torch.deg2rad(secondary_tilt_deg)
    rotation_matrix = torch.stack([-torch.sin(dt_m_sy), -torch.sin(sx) * torch.cos(dt_m_sy), torch.cos(sx) * torch.cos(dt_m_sy), torch.zeros_like(sx), torch.cos(sx), torch.sin(sx), -torch.cos(dt_m_sy), torch.sin(sx) * torch.sin(dt_m_sy), -torch.sin(dt_m_sy) * torch.cos(sx)], dim=-1).view(-1, 3, 3)
    tx = pcx * cols * torch.sin(sx) * torch.cos(dt_m_sy) + pcy * rows * torch.sin(dt_m_sy) + pcz * rows * torch.cos(sx) * torch.cos(dt_m_sy) - torch.sin(sx) * torch.cos(dt_m_sy) / 2 - torch.sin(dt_m_sy) / 2
    ty = -cols * pcx * torch.cos(sx) + pcz * rows * torch.sin(sx) + torch.cos(sx) / 2
    tz = -cols * pcx * torch.sin(sx) * torch.sin(dt_m_sy) + pcy * rows * torch.cos(dt_m_sy) - pcz * rows * torch.sin(dt_m_sy) * torch.cos(sx) + torch.sin(sx) * torch.sin(dt_m_sy) / 2 - torch.cos(dt_m_sy) / 2
    translation_vector = torch.stack([tx, ty, tz], dim=-1)
    return (rotation_matrix, translation_vector)

def get_radial_mask(shape: tuple[int, int], radius: Optional[float]=None, center: tuple[float, float]=(0.5, 0.5)) -> torch.Tensor:
    """
    Get a radial mask.

    Args:
        shape (tuple[int, int]): The shape of the mask.
        radius (float): The radius of the mask.
        center (tuple[float, float]): The center of the mask in fractional coordinates.

    Returns:
        torch.Tensor: The radial mask.

    """
    h, w = shape
    if radius is None:
        radius = min(h, w) / 2.0
    ii, jj = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing='ij')
    mask = (ii - center[0] * (h - 1)) ** 2 + (jj - center[1] * (w - 1)) ** 2 < radius ** 2
    return mask.to(torch.bool)
PROJECTION_CENTER=(.4221,.2179,.4954)
DETECTOR_SHAPE=(30,30)
SCAN_SHAPE=(40,50)
SCAN_PITCH_UM=1.5
def detector_rays(projection_center=PROJECTION_CENTER):
    """Sample-frame rays in detector-pixel-length units, exact integer0..29 centers.

    Qualified physical endpoint correction of the original linspace0..30.
    Bruker PC fractions and independently fixed70deg sample tilt are unchanged.
    """
    R,t=bruker_geometry_to_SE3(torch.tensor(projection_center).view(1,3),torch.tensor(-70.),torch.tensor(0.),(30,30))
    ij=torch.stack(torch.meshgrid(torch.arange(30.),torch.arange(30.),indexing='ij'),-1).reshape(-1,2)
    points=torch.cat([ij,torch.zeros_like(ij[:,:1])],-1)
    return ((R@points[...,None])+t[...,None]).squeeze(-1)
def interpolate_master(master,rays,align_corners=False,normalize_coords=True):
    """Explicit square-Lambert map then bilinear interpolation, border padding.

    master float32(401,802): north left, south right. Grid shape is a memory
    layout only; same independent bilinear values as original pseudo factoring.
    """
    if normalize_coords:rays=rays/rays.norm(dim=-1,keepdim=True)
    uv=rosca_lambert_side_by_side(rays);width=uv.shape[-2]
    grid=uv.reshape(1,-1,width,2)
    return torch.nn.functional.grid_sample(master[None,None],grid,mode='bilinear',padding_mode='border',align_corners=align_corners).reshape(rays.shape[:-1])
def simulate_patterns(master,quaternions,rays=None,align_corners=False):
    """Intensity samples after local q*(0,ray)*q-conjugate crystal rotation."""
    from .orientations import qu_apply
    if rays is None:rays=detector_rays()
    return interpolate_master(master,qu_apply(quaternions[...,None,:],rays),align_corners,True)
