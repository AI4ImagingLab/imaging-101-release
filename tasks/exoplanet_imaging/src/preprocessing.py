"""Preprocessing for ADI exoplanet imaging.

Loads raw_data.npz and applies circular masking and mean subtraction to
prepare the image cube for KLIP PSF subtraction.
"""

import numpy as np
import torch


def load_raw_data(npz_path: str):
    """Load observation data from raw_data.npz.

    Parameters
    ----------
    npz_path : str
        Path to raw_data.npz containing keys 'cube', 'angles', 'psf'.

    Returns
    -------
    cube : np.ndarray, shape (N, H, W), float32
        ADI image cube with N temporal frames.
    angles : np.ndarray, shape (N,), float32
        Parallactic angles in degrees for each frame.
    psf : np.ndarray, shape (Hpsf, Wpsf), float32
        Normalized off-axis PSF (used for FWHM estimation).
    """
    d = np.load(npz_path)
    cube = d['cube'][0].astype(np.float32)    # (1, N, H, W) → (N, H, W)
    angles = d['angles'][0].astype(np.float32)  # (1, N) → (N,)
    psf = d['psf'][0].astype(np.float32)        # (1, Hpsf, Wpsf) → (Hpsf, Wpsf)
    return cube, angles, psf


def create_circular_mask(
    ny: int,
    nx: int,
    center,
    iwa: float,
    owa: float = None,
    device: str = 'cpu',
) -> torch.Tensor:
    """Create a 2D boolean mask for the inner (and optionally outer) working angle.

    Parameters
    ----------
    ny, nx : int
        Image dimensions in pixels.
    center : tuple of (cx, cy)
        Center pixel coordinates (x, y).
    iwa : float
        Inner working angle radius in pixels.  Pixels with distance < iwa
        are masked (set to True).
    owa : float, optional
        Outer working angle radius in pixels.  Pixels with distance > owa
        are also masked.
    device : str
        Torch device.

    Returns
    -------
    mask : torch.BoolTensor, shape (ny, nx)
        True where pixels should be masked (set to NaN).
    """
    cx, cy = center
    y, x = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing='ij',
    )
    dist2 = (x - cx) ** 2 + (y - cy) ** 2
    mask = dist2 <= iwa ** 2
    if owa is not None:
        mask = mask | (dist2 >= owa ** 2)
    return mask


def apply_circular_mask(
    cube: np.ndarray,
    center,
    iwa: float,
    owa: float = None,
) -> np.ndarray:
    """Apply an inner (and outer) working angle mask to the cube.

    Pixels within `iwa` (and beyond `owa`) are set to NaN so they are
    excluded from PCA and SNR calculations.

    Parameters
    ----------
    cube : np.ndarray, shape (N, H, W)
    center : tuple (cx, cy)
    iwa : float   Inner working angle in pixels.
    owa : float, optional   Outer working angle in pixels.

    Returns
    -------
    masked_cube : np.ndarray, shape (N, H, W)
    """
    N, H, W = cube.shape
    t = torch.from_numpy(cube.copy())
    mask2d = create_circular_mask(H, W, center, iwa, owa)
    mask3d = mask2d.unsqueeze(0).expand(N, H, W)
    t.masked_fill_(mask3d, float('nan'))
    return t.numpy()


def mean_subtract_frames(cube: np.ndarray) -> np.ndarray:
    """Subtract the per-frame spatial mean (ignoring NaN) from each frame.

    Parameters
    ----------
    cube : np.ndarray, shape (N, H, W)

    Returns
    -------
    cube_ms : np.ndarray, shape (N, H, W)
    """
    flat = cube.reshape(cube.shape[0], -1)   # (N, H*W)
    means = np.nanmean(flat, axis=1, keepdims=True)  # (N, 1)
    return (flat - means).reshape(cube.shape)
