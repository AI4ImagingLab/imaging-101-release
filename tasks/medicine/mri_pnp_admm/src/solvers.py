"""
PnP-ADMM Solver for Compressed Sensing MRI
=============================================

Implements Plug-and-Play ADMM for CS-MRI reconstruction. The algorithm
alternates between a data-fidelity proximal step (solved in closed form
in the Fourier domain) and a denoising step (using a pretrained
RealSN-DnCNN as an implicit image prior).

ADMM iterations:
    1. v-update: proximal operator for ||M·F(x) - y||² / (2α)
       vf[sampled] = (La2·vf[sampled] + y[sampled]) / (1 + La2)
       where La2 = 1/(2α)
    2. x-update: denoise(2v - x_old - u_old) using RealSN-DnCNN
    3. u-update: u = u_old + x_old - v

The denoising step normalizes the input to [0,1], applies a scale/shift
to match the denoiser's training distribution, denoises, then inverts
the scale/shift and normalization.

Reference
---------
Ryu et al., "Plug-and-Play Methods Provably Converge with Properly
Trained Denoisers," ICML 2019.
"""

import numpy as np
import torch

from src.physics_model import simulate_observation, data_fidelity_proximal


def pnp_admm_reconstruct(
    model,
    im_orig: np.ndarray,
    mask: np.ndarray,
    noises: np.ndarray,
    alpha: float = 2.0,
    sigma: int = 15,
    maxitr: int = 100,
    device: str = "cpu",
) -> dict:
    """
    Reconstruct a 2D image from undersampled k-space using PnP-ADMM.

    Parameters
    ----------
    model : nn.Module
        Pretrained denoiser (residual: output = noise estimate).
    im_orig : ndarray, (m, n) float64
        Ground truth image (used to generate measurements and compute PSNR).
    mask : ndarray, (m, n) float64
        Binary k-space undersampling mask.
    noises : ndarray, (m, n) complex128
        Complex measurement noise.
    alpha : float
        ADMM penalty parameter.
    sigma : int
        Denoiser noise level (determines input scaling).
    maxitr : int
        Number of ADMM iterations.
    device : str
        Torch device for denoiser inference.

    Returns
    -------
    dict with keys:
        reconstruction : ndarray, (m, n) float64
            Final reconstructed image.
        zerofill : ndarray, (m, n) float64
            Zero-filled reconstruction (magnitude of IFFT of y).
        psnr_history : ndarray, (maxitr,) float64
            PSNR at each iteration.
        y_observed : ndarray, (m, n) complex128
            Observed k-space measurements.
    """
    m, n = im_orig.shape

    # Generate measurements
    y = simulate_observation(im_orig, mask, noises)
    x_init = np.fft.ifft2(y)
    zerofill = np.abs(x_init)

    # Initialize ADMM variables
    x = np.copy(zerofill)
    v = np.copy(x)
    u = np.zeros((m, n), dtype=np.float64)

    psnr_history = []

    for i in range(maxitr):
        xold = np.copy(x)
        uold = np.copy(u)

        # v-update: data fidelity proximal
        vtilde = x + u
        v = data_fidelity_proximal(vtilde, y, mask, alpha)

        # x-update: denoising step
        xtilde = 2 * v - xold - uold

        # Normalize to [0, 1]
        mintmp = np.min(xtilde)
        maxtmp = np.max(xtilde)
        xtilde_norm = (xtilde - mintmp) / (maxtmp - mintmp)

        # Scale/shift to match denoiser training distribution
        scale_range = 1.0 + sigma / 255.0 / 2.0
        scale_shift = (1 - scale_range) / 2.0
        xtilde_scaled = xtilde_norm * scale_range + scale_shift

        # Denoise with neural network
        xtilde_torch = torch.from_numpy(
            xtilde_scaled.reshape(1, 1, m, n)
        ).float().to(device)
        with torch.no_grad():
            r = model(xtilde_torch).cpu().numpy().reshape(m, n)
        x_denoised = xtilde_scaled - r

        # Invert scale/shift and normalization
        x = (x_denoised - scale_shift) / scale_range
        x = x * (maxtmp - mintmp) + mintmp

        # u-update: dual variable
        u = uold + xold - v

        # Track PSNR
        from src.visualization import compute_psnr
        psnr_history.append(compute_psnr(x, im_orig))

    return {
        "reconstruction": x,
        "zerofill": zerofill,
        "psnr_history": np.array(psnr_history),
        "y_observed": y,
    }
