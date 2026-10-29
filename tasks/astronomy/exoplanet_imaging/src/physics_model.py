"""Forward model for Angular Differential Imaging (ADI).

The ADI forward model describes how a static astrophysical scene A(x, y)
produces the observed image cube.  For a companion at sky position (r, PA),
frame k shows it at pixel position rotated by the parallactic angle θ_k
relative to the detector:

    T_k = I_ψ(n) + rot(A, -θ_k)(n)

where I_ψ is the quasi-static PSF speckle pattern, A is the sky scene, and
rot(·, θ) is a 2-D image rotation by θ degrees.

This module provides the rotation operator (used during derotation in the
solver) and the Karhunen-Loève (KL) basis construction that models the PSF
subspace.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Rotation (ADI forward/inverse model)
# ---------------------------------------------------------------------------

def rotate_frames(
    images: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """Rotate a stack of images by the given angles using a batched affine transform.

    Each image i is rotated by angles[i] degrees (counter-clockwise convention:
    positive angle rotates counter-clockwise).  This is implemented as a single
    batched bilinear grid_sample call on GPU or CPU.

    Parameters
    ----------
    images : torch.Tensor, shape (B, N, H, W)
        Batch of image cubes.  B is the number of K_klip values; N is the
        number of frames.
    angles : torch.Tensor, shape (N,)
        Rotation angles in degrees, one per frame.

    Returns
    -------
    rotated : torch.Tensor, shape (B, N, H, W)
    """
    B, N, H, W = images.shape
    flat = images.reshape(B * N, H, W).unsqueeze(1)  # (B*N, 1, H, W)
    repeated = angles.repeat(B)
    rad = torch.deg2rad(-repeated)  # negative for derotation convention
    cos_a = torch.cos(rad)
    sin_a = torch.sin(rad)
    theta = torch.zeros(B * N, 2, 3, device=images.device)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    grid = torch.nn.functional.affine_grid(theta, (B * N, 1, H, W), align_corners=False)
    rotated_flat = torch.nn.functional.grid_sample(flat, grid, align_corners=False)
    return rotated_flat.squeeze(1).reshape(B, N, H, W)


# ---------------------------------------------------------------------------
# KL basis construction (PSF subspace model)
# ---------------------------------------------------------------------------

def compute_kl_basis_svd(reference_flat: torch.Tensor, K_max: int) -> torch.Tensor:
    """Compute the KL basis via truncated SVD.

    Parameters
    ----------
    reference_flat : torch.Tensor, shape (N, n_pix)
        Mean-subtracted, NaN-replaced reference frame library.
    K_max : int
        Number of basis vectors to compute.

    Returns
    -------
    basis : torch.Tensor, shape (n_pix, K_max)
        Columns are the KL eigenvectors (right singular vectors).
    """
    _, _, Vt = torch.linalg.svd(reference_flat, full_matrices=False)
    return Vt.T[:, :K_max]


def compute_kl_basis_pca(reference_flat: torch.Tensor, K_max: int) -> torch.Tensor:
    """Compute the KL basis via PyTorch low-rank PCA.

    Parameters
    ----------
    reference_flat : torch.Tensor, shape (N, n_pix)
    K_max : int

    Returns
    -------
    basis : torch.Tensor, shape (n_pix, K_max)
    """
    _, _, V = torch.pca_lowrank(reference_flat, q=K_max, center=False)
    return V


def compute_kl_basis_eigh(reference_flat: torch.Tensor, K_max: int) -> torch.Tensor:
    """Compute the KL basis via eigendecomposition of the covariance matrix.

    Follows the original KLIP formulation (Soummer et al. 2012, Eq. 5):
    eigendecompose R R^T, then project back to pixel space.

    Parameters
    ----------
    reference_flat : torch.Tensor, shape (N, n_pix)
    K_max : int

    Returns
    -------
    basis : torch.Tensor, shape (n_pix, K_max)
    """
    cov = reference_flat @ reference_flat.T  # (N, N)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)
    scales = eigvals[:K_max].sqrt().reciprocal().unsqueeze(0)
    return reference_flat.T @ (eigvecs[:, :K_max] * scales)


_BASIS_METHODS = {
    'svd': compute_kl_basis_svd,
    'pca': compute_kl_basis_pca,
    'eigh': compute_kl_basis_eigh,
}


def compute_kl_basis(
    reference_flat: torch.Tensor,
    K_max: int,
    method: str = 'svd',
) -> torch.Tensor:
    """Dispatch to the selected KL basis computation method.

    Parameters
    ----------
    reference_flat : torch.Tensor, shape (N, n_pix)
    K_max : int
    method : {'svd', 'pca', 'eigh'}

    Returns
    -------
    basis : torch.Tensor, shape (n_pix, K_max)
    """
    if method not in _BASIS_METHODS:
        raise ValueError(f"method must be one of {list(_BASIS_METHODS)}, got {method!r}")
    return _BASIS_METHODS[method](reference_flat, K_max)
