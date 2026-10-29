"""Explicit POCA geometry, qualified from Muograph (AGPL-3.0).

Three disclosed source corrections preserve straight fitted lines, geometric
scattering angles and the cross-product nonparallel criterion. No upstream
package is imported. PyTorch supplies only general numerical primitives.
"""
import numpy as np
import torch

VOXEL_MIN_MM = (-180., -400., -860.)
VOXEL_MAX_MM = (180., 400., -460.)
VOXEL_WIDTH_MM = 10.
VOLUME_SHAPE = (36, 80, 40)


def fit_tracks(hits, corrected=True, chunk_size=50000):
    """Fit public float32(3,3,N) mm hits in float64; return directions and centroids (N,3).

    The principal right singular vector minimizes orthogonal line residuals.
    Correct orientation flips the entire vector when its z component is positive.
    The historical branch is retained solely for the explicit notebook audit.
    """
    h = torch.as_tensor(np.ascontiguousarray(hits), dtype=torch.float64)
    if h.ndim != 3 or h.shape[:2] != (3, 3) or h.shape[2] == 0:
        raise ValueError('Expected nonempty (3,3,N) detector hits')
    if not torch.isfinite(h).all():
        raise ValueError('Nonfinite hit')
    tracks = torch.empty((h.shape[2], 3), dtype=h.dtype)
    points = torch.empty_like(tracks)
    for start in range(0, h.shape[2], chunk_size):
        end = min(start + chunk_size, h.shape[2])
        block = h[:, :, start:end]
        center = block.mean(dim=1)
        centered = (block - center.unsqueeze(1)).permute(2, 1, 0)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        tracks[start:end] = vh[:, 0, :]
        points[start:end] = center.T
    if corrected:
        tracks = torch.where((tracks[:, 2] > 0).unsqueeze(1), -tracks, tracks)
    else:
        tracks[:, 2] = torch.where(tracks[:, 2] > 0, -tracks[:, 2], tracks[:, 2])
    return tracks.numpy(), points.numpy()


def scattering_angles(incoming, outgoing, corrected=True):
    """Stable float64(N,) geometric angular deflections in rad."""
    u = torch.as_tensor(incoming,dtype=torch.float64)
    v = torch.as_tensor(outgoing,dtype=torch.float64)
    if corrected:
        return torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(u,v,dim=1),dim=1),(u*v).sum(dim=1)).numpy()
    numerator=(u*v).abs().sum(dim=-1)
    denominator=torch.linalg.vector_norm(u,dim=1)*torch.linalg.vector_norm(v,dim=1)
    return torch.acos(torch.clamp(numerator/denominator,-1.,1.)).numpy()



def nonparallel_mask(incoming, outgoing, tol=1e-7, corrected=True):
    """Keep cross-product norm >=tol; historical componentwise cut is optional."""
    cross = torch.linalg.cross(torch.as_tensor(incoming,dtype=torch.float64), torch.as_tensor(outgoing,dtype=torch.float64), dim=1)
    mask = torch.linalg.vector_norm(cross, dim=1) >= tol if corrected else torch.all(cross.abs() >= tol, dim=1)
    return mask.numpy()


def closest_approach(points_in, points_out, tracks_in, tracks_out):
    """Midpoints of shortest skew-line connectors, float64(N,3), mm.

    Solve [u,-v,v脳u] t = q-p, then average p+t[0]u and q+t[1]v.
    Caller excludes parallel lines before this nonsingular three-equation solve.
    """
    p, q, u, v = [torch.as_tensor(a,dtype=torch.float64) for a in (points_in, points_out, tracks_in, tracks_out)]
    normal = torch.tensor(np.cross(v.numpy(), u.numpy()), dtype=torch.float64)
    matrix = torch.stack([u, -v, normal]).transpose(0, 1).transpose(-1, 1)
    parameters = torch.linalg.solve(matrix, q - p)
    left = p + parameters[:, 0, None] * u
    right = q + parameters[:, 1, None] * v
    return ((right - left) / 2 + left).numpy()


def in_volume(points):
    """Inclusive physical bounds mask (N,) for POCA coordinates in mm."""
    return np.all((points >= np.asarray(VOXEL_MIN_MM)) & (points <= np.asarray(VOXEL_MAX_MM)), axis=1)


def assign_voxels(points):
    """Return int32(N,3) indices with upper-cell ties and inclusive global max.

    Sorted-edge search evaluates the same interval predicates as the upstream
    exhaustive all-voxel comparisons without allocating N脳36脳80脳40 masks.
    On an internal shared edge the original last assignment selects the upper
    voxel. The uppermost physical edge remains in the final voxel.
    """
    if not in_volume(points).all():
        raise ValueError('POCA outside the declared volume')
    result = np.empty(points.shape, dtype=np.int32)
    for d, size in enumerate(VOLUME_SHAPE):
        edges = VOXEL_MIN_MM[d] + VOXEL_WIDTH_MM * np.arange(size + 1)
        result[:, d] = np.minimum(np.searchsorted(edges, points[:, d], side='right') - 1, size - 1)
    return result


def aggregate_rms(indices, angles, angle_quantile=.999, prediction_quantile=.999):
    """Per-voxel clipped angular RMS rad and int64 counts, both (36,80,40).

    Empty voxels are zero. The angle quantile uses retained in-volume events;
    the prediction quantile includes every voxel, preserving source conventions.
    """
    angle = torch.as_tensor(angles,dtype=torch.float64)
    if angle.numel() == 0:
        raise ValueError('No retained tracks')
    clipped = torch.clamp(angle, max=torch.quantile(angle, angle_quantile))
    idx = torch.as_tensor(indices, dtype=torch.int64)
    nx, ny, nz = VOLUME_SHAPE
    flat = idx[:, 0] * ny * nz + idx[:, 1] * nz + idx[:, 2]
    counts = torch.bincount(flat, minlength=nx * ny * nz)
    sums = torch.bincount(flat, weights=clipped ** 2, minlength=nx * ny * nz)
    rms = torch.zeros_like(sums)
    populated = counts > 0
    rms[populated] = torch.sqrt(sums[populated] / counts[populated])
    rms = rms.reshape(VOLUME_SHAPE)
    rms = torch.clamp(rms, max=torch.quantile(rms, prediction_quantile))
    return rms.numpy(), counts.reshape(VOLUME_SHAPE).numpy()
