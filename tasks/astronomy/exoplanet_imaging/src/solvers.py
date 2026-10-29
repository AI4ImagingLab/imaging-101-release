"""KLIP solver for ADI exoplanet direct imaging.

Implements the Karhunen-Loève Image Processing (KLIP) algorithm
(Soummer, Pueyo & Larkin 2012) for PSF subtraction in Angular Differential
Imaging (ADI) datasets.

Pipeline
--------
1. **PSF subtraction** (`compute_psf_residuals`): Build a KL basis from the full
   reference library and project each frame onto it; the residual is the
   PSF-subtracted frame.
2. **Derotation** (via `physics_model.rotate_frames`): Rotate each residual frame by
   -θ_k to align companions to a common sky orientation.
3. **Combination** (`combine_frames`): Take the mean or median across frames to
   produce the final detection map.
"""

from typing import Union, List

import numpy as np
import torch

from .preprocessing import apply_circular_mask, mean_subtract_frames
from .physics_model import compute_kl_basis, rotate_frames


# ---------------------------------------------------------------------------
# PSF subtraction
# ---------------------------------------------------------------------------

def compute_psf_residuals(
    cube: np.ndarray,
    K_klip: Union[int, List[int]],
    method: str = 'svd',
    device: str = 'cpu',
) -> torch.Tensor:
    """Compute PSF-subtracted residual cube for one or more KL truncation levels.

    For each frame, projects the mean-subtracted cube onto the truncated KL
    basis and subtracts the reconstruction.  All frames serve as the reference
    library (full-frame ADI KLIP, no annuli or subsections).

    Parameters
    ----------
    cube : np.ndarray, shape (N, H, W), float32
        Preprocessed (masked, NOT yet mean-subtracted) image cube.
    K_klip : int or list of int
        Number of KL modes to subtract.  Multiple values are computed in a
        single pass for efficiency.
    method : {'svd', 'pca', 'eigh'}
        KL basis computation method.
    device : str
        Torch device ('cpu' or 'cuda').

    Returns
    -------
    residuals : torch.Tensor, shape (n_K, N, H, W)
        PSF-subtracted residuals for each value of K_klip.
    """
    k_list = sorted([K_klip] if isinstance(K_klip, int) else list(K_klip))
    K_max = k_list[-1]
    N, H, W = cube.shape

    # Mean subtract and send to torch
    cube_ms = mean_subtract_frames(cube)
    data = torch.from_numpy(cube_ms.astype(np.float32)).to(device)
    ref_flat = torch.nan_to_num(data.view(N, -1), nan=0.0)  # (N, H*W)

    # Build KL basis from full reference library
    basis = compute_kl_basis(ref_flat, K_max, method)  # (H*W, K_max)

    residuals = torch.zeros(len(k_list), N, H, W, device=device)
    for i, k in enumerate(k_list):
        proj = ref_flat @ basis[:, :k]          # (N, k)
        ihat = proj @ basis[:, :k].T            # (N, H*W)
        residuals[i] = (ref_flat - ihat).view(N, H, W)

    # Restore NaN mask
    nan_mask = torch.isnan(data).unsqueeze(0)   # (1, N, H, W)
    residuals = residuals.masked_fill(nan_mask, float('nan'))
    return residuals


# ---------------------------------------------------------------------------
# Derotation and combination
# ---------------------------------------------------------------------------

def derotate_cube(
    residuals: torch.Tensor,
    angles: np.ndarray,
) -> torch.Tensor:
    """Derotate residual frames by their parallactic angles.

    Rotates each frame by -θ_k so that a companion at fixed sky position
    lines up across all frames.

    Parameters
    ----------
    residuals : torch.Tensor, shape (n_K, N, H, W)
    angles : np.ndarray, shape (N,)
        Parallactic angles in degrees.

    Returns
    -------
    derotated : torch.Tensor, shape (n_K, N, H, W)
    """
    ang_t = torch.tensor(angles.ravel().astype(np.float32), device=residuals.device)
    return rotate_frames(residuals, ang_t)


def combine_frames(
    derotated: torch.Tensor,
    statistic: str = 'mean',
) -> torch.Tensor:
    """Combine derotated frames along the temporal axis.

    Parameters
    ----------
    derotated : torch.Tensor, shape (n_K, N, H, W)
    statistic : {'mean', 'median'}

    Returns
    -------
    result : torch.Tensor, shape (n_K, H, W)
    """
    if statistic == 'mean':
        return torch.nanmean(derotated, dim=1)
    elif statistic == 'median':
        return torch.nanmedian(derotated, dim=1).values
    else:
        raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}")


# ---------------------------------------------------------------------------
# Full KLIP-ADI pipeline
# ---------------------------------------------------------------------------

def klip_adi(
    cube: np.ndarray,
    angles: np.ndarray,
    K_klip: Union[int, List[int]],
    iwa: float = None,
    center=None,
    method: str = 'svd',
    statistic: str = 'mean',
    device: str = 'cpu',
) -> np.ndarray:
    """Run the full KLIP+ADI pipeline on an image cube.

    Combines PSF subtraction, derotation, and frame combination into a
    single call.

    Parameters
    ----------
    cube : np.ndarray, shape (N, H, W)
        Raw ADI image cube (no prior masking required; masking is applied
        internally if `iwa` is given).
    angles : np.ndarray, shape (N,)
        Parallactic angles in degrees.
    K_klip : int or list of int
        KL truncation level(s).
    iwa : float, optional
        Inner working angle in pixels.  Pixels within this radius are set
        to NaN before PCA.
    center : tuple (cx, cy), optional
        Star center in pixels.  Defaults to image centre.
    method : {'svd', 'pca', 'eigh'}
    statistic : {'mean', 'median'}
    device : str

    Returns
    -------
    result : np.ndarray
        - shape (H, W) if K_klip is a single int
        - shape (n_K, H, W) if K_klip is a list
    """
    N, H, W = cube.shape
    if center is None:
        center = ((W - 1) / 2.0, (H - 1) / 2.0)

    # Apply IWA mask
    if iwa is not None:
        cube = apply_circular_mask(cube, center, iwa)

    is_single = isinstance(K_klip, int)
    k_list = [K_klip] if is_single else K_klip

    residuals = compute_psf_residuals(cube, k_list, method=method, device=device)
    derotated = derotate_cube(residuals, angles)
    result = combine_frames(derotated, statistic=statistic)  # (n_K, H, W)

    # Re-apply IWA mask to final image
    if iwa is not None:
        from .preprocessing import create_circular_mask
        mask2d = create_circular_mask(H, W, center, iwa, device=device)
        result = result.masked_fill(mask2d.unsqueeze(0), float('nan'))

    result_np = result.cpu().numpy()
    return result_np[0] if is_single else result_np
