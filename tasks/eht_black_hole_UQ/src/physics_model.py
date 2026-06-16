"""
Physics Model for DPI — GPU NUFFT Forward Model and Loss Functions
===================================================================

Implements the interferometric forward model using GPU-accelerated
Non-Uniform FFT (NUFFT) via torchkbnufft, plus closure phase / closure
amplitude computation and all data fidelity and regularization losses.

Reference
---------
Sun & Bouman (2020), arXiv:2010.14462 — Eqs. 7–16
Original code: DPI/DPItorch/interferometry_helpers.py
"""

import numpy as np
import torch
import torch.nn as nn
from torchkbnufft import KbNufft


# ── Complex arithmetic helpers ──────────────────────────────────────────────

def torch_complex_mul(x, y):
    """
    Element-wise complex multiplication using real-valued tensors.

    Parameters
    ----------
    x : (B, 2, M) tensor — [real; imag] channels
    y : (2, M) tensor — [real; imag] channels

    Returns
    -------
    (B, 2, M) tensor — complex product
    """
    xy_real = x[:, :, 0:1] * y[0:1] - x[:, :, 1::] * y[1::]
    xy_imag = x[:, :, 0:1] * y[1::] + x[:, :, 1::] * y[0:1]
    return torch.cat([xy_real, xy_imag], -2)


def torch_complex_matmul(x, F):
    """
    Complex matrix-vector multiply using real-valued tensors.

    Parameters
    ----------
    x : (B, N²) tensor — real-valued flattened images
    F : (N², M, 2) tensor — DFT matrix [real, imag]

    Returns
    -------
    (B, 2, M) tensor — complex visibilities
    """
    Fx_real = torch.matmul(x, F[:, :, 0])
    Fx_imag = torch.matmul(x, F[:, :, 1])
    return torch.cat([Fx_real.unsqueeze(1), Fx_imag.unsqueeze(1)], -2)


# ── NUFFT Forward Model ────────────────────────────────────────────────────

class NUFFTForwardModel(nn.Module):
    """
    GPU NUFFT forward model for VLBI interferometry.

    Computes complex visibilities, visibility amplitudes, closure phases,
    and log closure amplitudes from a batch of images.

    Parameters
    ----------
    npix : int
        Image size (pixels per side).
    ktraj_vis : (1, 2, M) Tensor
        Scaled (v, u) trajectory for torchkbnufft.
    pulsefac_vis : (2, M) Tensor
        Pulse function correction factor [real; imag].
    cphase_ind_list : list of 3 LongTensors
        Visibility indices for closure phase triangles.
    cphase_sign_list : list of 3 FloatTensors
        Conjugation signs for closure phases.
    camp_ind_list : list of 4 LongTensors
        Visibility indices for closure amplitude quadrangles.
    device : torch.device
        Computation device (CPU or CUDA).
    """

    def __init__(self, npix, ktraj_vis, pulsefac_vis,
                 cphase_ind_list, cphase_sign_list, camp_ind_list,
                 device):
        super().__init__()
        self.npix = npix
        self.device = device
        self.eps = 1e-16

        self.nufft_ob = KbNufft(im_size=(npix, npix), numpoints=3).to(device)
        self.ktraj_vis = ktraj_vis.to(device)
        self.pulsefac_vis = pulsefac_vis.to(device)

        self.cphase_ind = [ind.to(device) for ind in cphase_ind_list]
        self.cphase_sign = [s.to(device) for s in cphase_sign_list]
        self.camp_ind = [ind.to(device) for ind in camp_ind_list]

    def forward(self, images):
        """
        Compute visibilities and closure quantities from a batch of images.

        Parameters
        ----------
        images : (B, npix, npix) Tensor
            Batch of non-negative images.

        Returns
        -------
        vis : (B, 2, M) Tensor — complex visibilities [real; imag]
        visamp : (B, M) Tensor — visibility amplitudes
        cphase : (B, N_cp) Tensor — closure phases in degrees
        logcamp : (B, N_ca) Tensor — log closure amplitudes
        """
        npix = self.npix
        eps = self.eps

        # Reshape for torchkbnufft: (B, npix, npix) → (1, B, npix, npix, 2)
        x = images.reshape(-1, npix, npix).type(torch.float32).to(self.device)
        x = x.unsqueeze(-1)
        x = torch.cat([x, torch.zeros_like(x)], -1)
        x = x.unsqueeze(0)

        kdata = self.nufft_ob(x, self.ktraj_vis)
        kdata = kdata.transpose(-1, -2)
        vis = torch_complex_mul(kdata, self.pulsefac_vis).squeeze(0)

        # Visibility amplitude
        visamp = torch.sqrt(vis[:, 0, :] ** 2 + vis[:, 1, :] ** 2 + eps)

        # Closure phases
        vis1 = torch.index_select(vis, -1, self.cphase_ind[0])
        vis2 = torch.index_select(vis, -1, self.cphase_ind[1])
        vis3 = torch.index_select(vis, -1, self.cphase_ind[2])

        ang1 = torch.atan2(vis1[:, 1, :], vis1[:, 0, :])
        ang2 = torch.atan2(vis2[:, 1, :], vis2[:, 0, :])
        ang3 = torch.atan2(vis3[:, 1, :], vis3[:, 0, :])
        cphase = (self.cphase_sign[0] * ang1 +
                  self.cphase_sign[1] * ang2 +
                  self.cphase_sign[2] * ang3) * 180 / np.pi

        # Log closure amplitudes
        vis12 = torch.index_select(vis, -1, self.camp_ind[0])
        vis12_amp = torch.sqrt(vis12[:, 0, :] ** 2 + vis12[:, 1, :] ** 2 + eps)
        vis34 = torch.index_select(vis, -1, self.camp_ind[1])
        vis34_amp = torch.sqrt(vis34[:, 0, :] ** 2 + vis34[:, 1, :] ** 2 + eps)
        vis14 = torch.index_select(vis, -1, self.camp_ind[2])
        vis14_amp = torch.sqrt(vis14[:, 0, :] ** 2 + vis14[:, 1, :] ** 2 + eps)
        vis23 = torch.index_select(vis, -1, self.camp_ind[3])
        vis23_amp = torch.sqrt(vis23[:, 0, :] ** 2 + vis23[:, 1, :] ** 2 + eps)

        logcamp = (torch.log(vis12_amp) + torch.log(vis34_amp)
                   - torch.log(vis14_amp) - torch.log(vis23_amp))

        return vis, visamp, cphase, logcamp


# ── Data fidelity loss functions ────────────────────────────────────────────

def Loss_angle_diff(sigma, device):
    """
    Closure phase chi-squared loss: 2 * mean((1 - cos(true - pred)) / sigma^2).

    Parameters
    ----------
    sigma : array-like — closure phase noise std (degrees)
    device : torch.device
    """
    sigma = torch.Tensor(sigma).type(torch.float32).to(device=device)

    def func(y_true, y_pred):
        angle_true = y_true * np.pi / 180
        angle_pred = y_pred * np.pi / 180
        return 2.0 * torch.mean(
            (1 - torch.cos(angle_true - angle_pred)) / (sigma * np.pi / 180) ** 2, 1)
    return func


def Loss_logca_diff2(sigma, device):
    """
    Log closure amplitude chi-squared loss: mean((true - pred)^2 / sigma^2).

    Parameters
    ----------
    sigma : array-like — log closure amplitude noise std
    device : torch.device
    """
    sigma = torch.Tensor(sigma).type(torch.float32).to(device=device)

    def func(y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2 / sigma ** 2, 1)
    return func


def Loss_vis_diff(sigma, device):
    """
    Complex visibility chi-squared loss.

    Parameters
    ----------
    sigma : array-like — visibility noise std
    device : torch.device
    """
    sigma = torch.Tensor(sigma).type(torch.float32).to(device=device)

    def func(y_true, y_pred):
        return torch.mean(
            ((y_true[0] - y_pred[:, 0]) ** 2 + (y_true[1] - y_pred[:, 1]) ** 2) / sigma ** 2, 1)
    return func


def Loss_logamp_diff(sigma, device):
    """
    Log visibility amplitude chi-squared loss.

    Parameters
    ----------
    sigma : array-like — visibility noise std
    device : torch.device
    """
    sigma = torch.Tensor(sigma).type(torch.float32).to(device=device)

    def func(y_true, y_pred):
        return torch.mean(
            y_true ** 2 / sigma ** 2 * (torch.log(y_true) - torch.log(y_pred)) ** 2, 1)
    return func


def Loss_visamp_diff(sigma, device):
    """Visibility amplitude squared difference loss."""
    sigma = torch.Tensor(sigma).type(torch.float32).to(device=device)

    def func(y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2 / sigma ** 2, 1)
    return func


# ── Image prior loss functions ──────────────────────────────────────────────

def Loss_l1(y_pred):
    """L1 sparsity prior: mean(|image|)."""
    return torch.mean(torch.abs(y_pred), (-1, -2))


def Loss_TSV(y_pred):
    """Total Squared Variation prior."""
    return (torch.mean((y_pred[:, 1::, :] - y_pred[:, 0:-1, :]) ** 2, (-1, -2)) +
            torch.mean((y_pred[:, :, 1::] - y_pred[:, :, 0:-1]) ** 2, (-1, -2)))


def Loss_TV(y_pred):
    """Total Variation prior."""
    return (torch.mean(torch.abs(y_pred[:, 1::, :] - y_pred[:, 0:-1, :]), (-1, -2)) +
            torch.mean(torch.abs(y_pred[:, :, 1::] - y_pred[:, :, 0:-1]), (-1, -2)))


def Loss_flux(flux):
    """Flux conservation constraint: (sum(image) - flux)^2."""
    def func(y_pred):
        return (torch.sum(y_pred, (-1, -2)) - flux) ** 2
    return func


def Loss_center(device, center=15.5, dim=32):
    """
    Centering constraint: penalizes center-of-mass deviation from image center.

    Parameters
    ----------
    device : torch.device
    center : float — target center coordinate (default: dim/2 - 0.5)
    dim : int — image dimension
    """
    X = np.concatenate([np.arange(dim).reshape((1, dim))] * dim, 0)
    Y = np.concatenate([np.arange(dim).reshape((dim, 1))] * dim, 1)
    X = torch.Tensor(X).type(torch.float32).to(device=device)
    Y = torch.Tensor(Y).type(torch.float32).to(device=device)

    def func(y_pred):
        y_pred_flux = torch.mean(y_pred, (-1, -2))
        xc_pred_norm = torch.mean(y_pred * X, (-1, -2)) / y_pred_flux
        yc_pred_norm = torch.mean(y_pred * Y, (-1, -2)) / y_pred_flux
        loss = 0.5 * ((xc_pred_norm - center) ** 2 + (yc_pred_norm - center) ** 2)
        return loss
    return func


def Loss_cross_entropy(y_true, y_pred):
    """
    Maximum entropy (MEM) cross-entropy prior.

    Measures KL-divergence between reconstructed image and prior image.
    """
    return torch.mean(
        y_pred * (torch.log(y_pred + 1e-12) - torch.log(y_true + 1e-12)), (-1, -2))
