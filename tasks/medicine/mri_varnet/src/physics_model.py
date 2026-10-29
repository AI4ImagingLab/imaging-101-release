"""
Multi-Coil Cartesian MRI Forward Model for VarNet
====================================================

Forward model: y_c = M * F(S_c * x) for each coil c

VarNet jointly estimates sensitivity maps and performs iterative
reconstruction using an unrolled optimization network.

This module provides utility functions for the forward/adjoint
operations used internally by VarNet.

Centered FFT and RSS ported from fastMRI (facebookresearch/fastMRI):
    fastmri/fftc.py  (fft2c_new, ifft2c_new)
    fastmri/coil_combine.py  (rss)
"""

import numpy as np
import torch
from typing import List, Optional


# ---------------------------------------------------------------------------
# Centered FFT helpers (ported from fastmri/fftc.py)
# ---------------------------------------------------------------------------

def _roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """Roll tensor along one dimension (circular shift)."""
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def _roll(x: torch.Tensor, shift: List[int], dim: List[int]) -> torch.Tensor:
    """Roll tensor along multiple dimensions."""
    for (s, d) in zip(shift, dim):
        x = _roll_one_dim(x, s, d)
    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """FFT shift for torch tensors."""
    if dim is None:
        dim = list(range(x.dim()))
    shift = [x.shape[d] // 2 for d in dim]
    return _roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """Inverse FFT shift for torch tensors."""
    if dim is None:
        dim = list(range(x.dim()))
    shift = [(x.shape[d] + 1) // 2 for d in dim]
    return _roll(x, shift, dim)


def fft2c(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Centered 2D FFT on torch tensor with real-valued last dim of size 2.

    Ported from fastMRI fftc.py fft2c_new().

    Parameters
    ----------
    data : Tensor, (..., H, W, 2)
        Input in real-valued format (last dim = [real, imag]).

    Returns
    -------
    Tensor, (..., H, W, 2)
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(torch.view_as_complex(data), dim=(-2, -1), norm=norm)
    )
    data = fftshift(data, dim=[-3, -2])
    return data


def ifft2c(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Centered 2D inverse FFT on torch tensor with real-valued last dim of size 2.

    Ported from fastMRI fftc.py ifft2c_new().

    Parameters
    ----------
    data : Tensor, (..., H, W, 2)
        Input in real-valued format (last dim = [real, imag]).

    Returns
    -------
    Tensor, (..., H, W, 2)
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(torch.view_as_complex(data), dim=(-2, -1), norm=norm)
    )
    data = fftshift(data, dim=[-3, -2])
    return data


# ---------------------------------------------------------------------------
# Root-sum-of-squares (ported from fastmri/coil_combine.py)
# ---------------------------------------------------------------------------

def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Root Sum of Squares coil combination.

    Ported from fastMRI coil_combine.py rss().

    Parameters
    ----------
    data : Tensor
        Complex-valued input (torch.complex).
    dim : int
        Coil dimension to reduce.

    Returns
    -------
    Tensor (real-valued)
    """
    return torch.sqrt((data.abs() ** 2).sum(dim))


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def center_crop(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop an image to target size."""
    h, w = img.shape[-2:]
    sh = (h - target_h) // 2
    sw = (w - target_w) // 2
    return img[..., sh:sh + target_h, sw:sw + target_w]


def zero_filled_recon(masked_kspace: torch.Tensor) -> np.ndarray:
    """
    Zero-filled reconstruction: IFFT + RSS coil combination.

    Parameters
    ----------
    masked_kspace : Tensor, (Nc, H, W, 2) float32

    Returns
    -------
    recon : ndarray, (H, W) float32
    """
    zf_complex = ifft2c(masked_kspace)
    zf_view = torch.view_as_complex(zf_complex)
    return rss(zf_view, dim=0).numpy()
