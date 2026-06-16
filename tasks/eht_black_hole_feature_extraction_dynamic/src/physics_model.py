"""
Physics Model for α-DPI — Geometric Models and NUFFT Forward Model
===================================================================

Implements geometric black hole image models (crescent + elliptical Gaussians)
and the NUFFT-based interferometric forward model for computing closure
quantities from images.

Reference
---------
Sun et al. (2022), ApJ 932:99 — α-DPI
Original code: DPI/DPItorch/geometric_model.py
               DPI/DPItorch/interferometry_helpers.py
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


# ── Geometric Image Models ─────────────────────────────────────────────────

class SimpleCrescentParam2Img(nn.Module):
    """
    Simple crescent (asymmetric Gaussian ring) model.

    Maps 4 unit-interval parameters [r, sigma, s, eta] to a 2D image:
      - r: ring radius (mapped to r_range in uas)
      - sigma: ring width (mapped to width_range in uas)
      - s: brightness asymmetry [0, 1]
      - eta: position angle (mapped to [-181, 181] degrees)

    The image is a Gaussian ring modulated by (1 + s * cos(theta - eta)),
    normalized to unit total flux.

    Parameters
    ----------
    npix : int
        Image size in pixels.
    fov : float
        Field of view in microarcseconds.
    r_range : list
        [min, max] radius in microarcseconds.
    width_range : list
        [min, max] width in microarcseconds.
    """

    def __init__(self, npix, fov=120, r_range=[10.0, 40.0],
                 width_range=[1.0, 40.0]):
        super().__init__()
        self.fov = fov
        self.r_range = r_range
        self.width_range = width_range
        self.nparams = 4
        self.eps = 1e-4
        self.gap = 1.0 / npix
        xs = torch.arange(-1 + self.gap, 1, 2 * self.gap)
        grid_y, grid_x = torch.meshgrid(-xs, xs, indexing='ij')
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_r = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        self.grid_theta = torch.atan2(grid_y, grid_x)

    def compute_features(self, params):
        """
        Convert unit-interval parameters to physical features.

        Parameters
        ----------
        params : (B, 4) tensor — parameters in [0, 1]

        Returns
        -------
        r, sigma, s, eta : tensors of shape (B, 1, 1) — physical parameters
        """
        half_fov = 0.5 * self.fov
        r = self.r_range[0] / half_fov + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1] - self.r_range[0]) / half_fov
        sigma = self.width_range[0] / half_fov + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1] - self.width_range[0]) / half_fov
        s = params[:, 2].unsqueeze(-1).unsqueeze(-1)
        eta = 181 / 180 * np.pi * (2.0 * params[:, 3].unsqueeze(-1).unsqueeze(-1) - 1.0)
        return r, sigma, s, eta

    def forward(self, params):
        """
        Generate crescent images from parameters.

        Parameters
        ----------
        params : (B, 4) tensor — parameters in [0, 1]

        Returns
        -------
        (B, npix, npix) tensor — normalized crescent images
        """
        r, sigma, s, eta = self.compute_features(params)
        ring = torch.exp(-0.5 * (self.grid_r - r) ** 2 / sigma ** 2)
        S = 1 + s * torch.cos(self.grid_theta - eta)
        crescent = S * ring
        crescent = crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1) + self.eps)
        return crescent

    def to(self, device):
        self.grid_x = self.grid_x.to(device)
        self.grid_y = self.grid_y.to(device)
        self.grid_r = self.grid_r.to(device)
        self.grid_theta = self.grid_theta.to(device)
        return self


class SimpleCrescentNuisanceParam2Img(nn.Module):
    """
    Crescent + N elliptical Gaussian model.

    Extends SimpleCrescentParam2Img with N nuisance Gaussian components.
    Each Gaussian adds 6 parameters: (x, y, scale, sigma_x, sigma_y, rho).

    Total parameters: 4 + 6 * n_gaussian

    Parameters
    ----------
    npix : int
        Image size in pixels.
    n_gaussian : int
        Number of nuisance Gaussian components.
    fov : float
        Field of view in microarcseconds.
    r_range : list
        [min, max] radius in microarcseconds.
    width_range : list
        [min, max] width in microarcseconds.
    """

    def __init__(self, npix, n_gaussian=1, fov=120, r_range=[10.0, 40.0],
                 width_range=[1.0, 40.0]):
        super().__init__()
        self.n_gaussian = n_gaussian
        self.fov = fov
        self.r_range = r_range
        self.width_range = width_range
        self.nparams = 4 + 6 * n_gaussian
        self.eps = 1e-4
        self.gap = 1.0 / npix
        xs = torch.arange(-1 + self.gap, 1, 2 * self.gap)
        grid_y, grid_x = torch.meshgrid(-xs, xs, indexing='ij')
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_r = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        self.grid_theta = torch.atan2(grid_y, grid_x)

    def compute_features(self, params):
        """
        Convert unit-interval parameters to physical features.

        Parameters
        ----------
        params : (B, nparams) tensor — parameters in [0, 1]

        Returns
        -------
        tuple: (r, sigma, s, eta, nuisance_scale, nuisance_x, nuisance_y,
                nuisance_covinv1, nuisance_covinv2, nuisance_covinv12)
        """
        half_fov = 0.5 * self.fov
        r = self.r_range[0] / half_fov + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1] - self.r_range[0]) / half_fov
        sigma = self.width_range[0] / half_fov + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1] - self.width_range[0]) / half_fov
        s = params[:, 2].unsqueeze(-1).unsqueeze(-1)
        eta = 181 / 180 * np.pi * (2.0 * params[:, 3].unsqueeze(-1).unsqueeze(-1) - 1.0)

        nuisance_scale = []
        nuisance_covinv1 = []
        nuisance_covinv2 = []
        nuisance_covinv12 = []
        nuisance_x = []
        nuisance_y = []
        for k in range(self.n_gaussian):
            x_shift = (2 * params[:, 4 + k * 6] - 1).unsqueeze(-1).unsqueeze(-1)
            y_shift = (2 * params[:, 5 + k * 6] - 1).unsqueeze(-1).unsqueeze(-1)
            scale = params[:, 6 + k * 6].unsqueeze(-1).unsqueeze(-1)
            sigma_x = params[:, 7 + k * 6].unsqueeze(-1).unsqueeze(-1)
            sigma_y = params[:, 8 + k * 6].unsqueeze(-1).unsqueeze(-1)
            rho = 2 * 0.99 * (params[:, 9 + k * 6].unsqueeze(-1).unsqueeze(-1) - 0.5)
            sigma_xy = rho * sigma_x * sigma_y
            factor = self.eps ** 2 + self.eps * sigma_x ** 2 + self.eps * sigma_y ** 2 + (1 - rho ** 2) * sigma_x ** 2 * sigma_y ** 2
            covinv1 = sigma_y ** 2 / factor
            covinv2 = sigma_x ** 2 / factor
            covinv12 = sigma_xy / factor

            nuisance_x.append(x_shift)
            nuisance_y.append(y_shift)
            nuisance_scale.append(scale)
            nuisance_covinv1.append(covinv1)
            nuisance_covinv2.append(covinv2)
            nuisance_covinv12.append(covinv12)

        return r, sigma, s, eta, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12

    def forward(self, params):
        """
        Generate crescent + Gaussians images from parameters.

        Parameters
        ----------
        params : (B, nparams) tensor — parameters in [0, 1]

        Returns
        -------
        (B, npix, npix) tensor — normalized images
        """
        r, sigma, s, eta, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12 = self.compute_features(params)

        ring = torch.exp(-0.5 * (self.grid_r - r) ** 2 / sigma ** 2)
        S = 1 + s * torch.cos(self.grid_theta - eta)
        crescent = S * ring

        for k in range(self.n_gaussian):
            x_c = self.grid_x - nuisance_x[k]
            y_c = self.grid_y - nuisance_y[k]
            delta = 0.5 * (nuisance_covinv1[k] * x_c ** 2 + nuisance_covinv2[k] * y_c ** 2 - 2 * nuisance_covinv12[k] * x_c * y_c)
            nuisance_now = torch.exp(-delta) * nuisance_scale[k]
            crescent += nuisance_now

        crescent = crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1) + self.eps)
        return crescent

    def to(self, device):
        self.grid_x = self.grid_x.to(device)
        self.grid_y = self.grid_y.to(device)
        self.grid_r = self.grid_r.to(device)
        self.grid_theta = self.grid_theta.to(device)
        return self


class SimpleCrescentNuisanceFloorParam2Img(nn.Module):
    """
    Crescent + floor disk + N rotated elliptical Gaussians.

    This is the model used in Figure 7 of Sun et al. (2022).
    Compared to SimpleCrescentNuisanceParam2Img, this adds:
      - A tapered disk "floor" component blended with the crescent ring
      - A crescent_flux scaling factor
      - Gaussians parameterized by (x, y, scale, sigma_x, sigma_y, theta)
        with explicit shift_range and sigma_range

    Total parameters: 4 + 6 * n_gaussian + 2 (floor + crescent_flux)

    Parameters
    ----------
    npix : int
        Image size in pixels.
    n_gaussian : int
        Number of nuisance Gaussian components.
    fov : float
        Field of view in microarcseconds.
    r_range : list
        [min, max] radius in microarcseconds.
    width_range : list
        [min, max] width in microarcseconds.
    asym_range : list
        [min, max] asymmetry parameter.
    floor_range : list
        [min, max] floor fraction.
    crescent_flux_range : list
        [min, max] crescent flux scale.
    shift_range : list
        [min, max] Gaussian center offset in microarcseconds.
    sigma_range : list
        [min, max] Gaussian width in microarcseconds.
    gaussian_scale_range : list
        [min, max] Gaussian flux scale.
    """

    def __init__(self, npix, n_gaussian=2, fov=120,
                 r_range=None, width_range=None,
                 asym_range=None, floor_range=None,
                 crescent_flux_range=None,
                 shift_range=None, sigma_range=None,
                 gaussian_scale_range=None):
        super().__init__()
        self.n_gaussian = n_gaussian
        self.fov = fov
        self.r_range = r_range or [10.0, 40.0]
        self.width_range = width_range or [1.0, 40.0]
        self.asym_range = asym_range or [1e-3, 0.99]
        self.floor_range = floor_range or [0.0, 1.0]
        self.crescent_flux_range = crescent_flux_range or [1e-3, 2.0]
        self.shift_range = shift_range or [-200.0, 200.0]
        self.sigma_range = sigma_range or [1.0, 100.0]
        self.gaussian_scale_range = gaussian_scale_range or [1e-3, 2.0]

        # 4 crescent + 6 per Gaussian + 2 (floor, crescent_flux)
        self.nparams = 4 + 6 * n_gaussian + 2

        self.eps = 1e-4
        self.gap = 1.0 / npix
        xs = torch.arange(-1 + self.gap, 1, 2 * self.gap)
        grid_y, grid_x = torch.meshgrid(-xs, xs, indexing='ij')
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_r = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        self.grid_theta = torch.atan2(grid_y, grid_x)
        self.npix = npix

    def compute_features(self, params):
        half_fov = 0.5 * self.fov
        r = self.r_range[0] / half_fov + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1] - self.r_range[0]) / half_fov
        sigma = self.width_range[0] / half_fov + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1] - self.width_range[0]) / half_fov
        s = self.asym_range[0] + params[:, 2].unsqueeze(-1).unsqueeze(-1) * (self.asym_range[1] - self.asym_range[0])
        eta = 181 / 180 * np.pi * (2.0 * params[:, 3].unsqueeze(-1).unsqueeze(-1) - 1.0)

        nuisance_scale = []
        sigma_x_list = []
        sigma_y_list = []
        theta_list = []
        nuisance_x = []
        nuisance_y = []
        for k in range(self.n_gaussian):
            x_shift = self.shift_range[0] / half_fov + params[:, 4 + k * 6].unsqueeze(-1).unsqueeze(-1) * (self.shift_range[1] - self.shift_range[0]) / half_fov
            y_shift = self.shift_range[0] / half_fov + params[:, 5 + k * 6].unsqueeze(-1).unsqueeze(-1) * (self.shift_range[1] - self.shift_range[0]) / half_fov
            scale = self.gaussian_scale_range[0] + params[:, 6 + k * 6].unsqueeze(-1).unsqueeze(-1) * (self.gaussian_scale_range[1] - self.gaussian_scale_range[0])
            sx = self.sigma_range[0] / half_fov + params[:, 7 + k * 6].unsqueeze(-1).unsqueeze(-1) * (self.sigma_range[1] - self.sigma_range[0]) / half_fov
            sy = self.sigma_range[0] / half_fov + params[:, 8 + k * 6].unsqueeze(-1).unsqueeze(-1) * (self.sigma_range[1] - self.sigma_range[0]) / half_fov
            theta = 181 / 180 * 0.5 * np.pi * params[:, 9 + k * 6].unsqueeze(-1).unsqueeze(-1)

            nuisance_x.append(x_shift)
            nuisance_y.append(y_shift)
            nuisance_scale.append(scale)
            sigma_x_list.append(sx)
            sigma_y_list.append(sy)
            theta_list.append(theta)

        # Floor and crescent_flux: last 2 params
        floor = self.floor_range[0] + (self.floor_range[1] - self.floor_range[0]) * params[:, 4 + self.n_gaussian * 6].unsqueeze(-1).unsqueeze(-1)
        crescent_flux = self.crescent_flux_range[0] + (self.crescent_flux_range[1] - self.crescent_flux_range[0]) * params[:, 5 + self.n_gaussian * 6].unsqueeze(-1).unsqueeze(-1)

        return (r, sigma, s, eta, crescent_flux, floor,
                nuisance_scale, nuisance_x, nuisance_y,
                sigma_x_list, sigma_y_list, theta_list)

    def forward(self, params):
        (r, sigma, s, eta, crescent_flux, floor,
         nuisance_scale, nuisance_x, nuisance_y,
         sigma_x_list, sigma_y_list, theta_list) = self.compute_features(params)

        # Asymmetric ring
        ring = torch.exp(-0.5 * (self.grid_r - r) ** 2 / sigma ** 2)
        S = 1 + s * torch.cos(self.grid_theta - eta)
        crescent = S * ring

        # Tapered disk (floor)
        disk = 0.5 * (1 + torch.erf((r - self.grid_r) / (np.sqrt(2) * sigma)))

        # Normalize and blend
        crescent = crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1) + self.eps)
        disk = disk / (torch.sum(disk, (-1, -2)).unsqueeze(-1).unsqueeze(-1) + self.eps)
        crescent = crescent_flux * ((1 - floor) * crescent + floor * disk)

        # Add Gaussians
        for k in range(self.n_gaussian):
            x_c = self.grid_x - nuisance_x[k]
            y_c = self.grid_y - nuisance_y[k]
            x_rot = x_c * torch.cos(theta_list[k]) + y_c * torch.sin(theta_list[k])
            y_rot = -x_c * torch.sin(theta_list[k]) + y_c * torch.cos(theta_list[k])
            delta = 0.5 * (x_rot ** 2 / sigma_x_list[k] ** 2 + y_rot ** 2 / sigma_y_list[k] ** 2)
            nuisance_now = 1 / (2 * np.pi * sigma_x_list[k] * sigma_y_list[k]) * torch.exp(-delta)
            nuisance_now = nuisance_now / (torch.sum(nuisance_now, (-1, -2)).unsqueeze(-1).unsqueeze(-1) + self.eps)
            nuisance_now = nuisance_scale[k] * nuisance_now
            crescent += nuisance_now

        # Final normalization
        crescent = crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1) + self.eps)

        return crescent

    def to(self, device):
        self.grid_x = self.grid_x.to(device)
        self.grid_y = self.grid_y.to(device)
        self.grid_r = self.grid_r.to(device)
        self.grid_theta = self.grid_theta.to(device)
        return self


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

        Returns
        -------
        vis : (B, 2, M) Tensor — complex visibilities [real; imag]
        visamp : (B, M) Tensor — visibility amplitudes
        cphase : (B, N_cp) Tensor — closure phases in degrees
        logcamp : (B, N_ca) Tensor — log closure amplitudes
        """
        npix = self.npix
        eps = self.eps

        x = images.reshape(-1, npix, npix).type(torch.float32).to(self.device)
        x = x.unsqueeze(-1)
        x = torch.cat([x, torch.zeros_like(x)], -1)
        x = x.unsqueeze(0)

        kdata = self.nufft_ob(x, self.ktraj_vis)
        kdata = kdata.transpose(-1, -2)
        vis = torch_complex_mul(kdata, self.pulsefac_vis).squeeze(0)

        visamp = torch.sqrt(vis[:, 0, :] ** 2 + vis[:, 1, :] ** 2 + eps)

        vis1 = torch.index_select(vis, -1, self.cphase_ind[0])
        vis2 = torch.index_select(vis, -1, self.cphase_ind[1])
        vis3 = torch.index_select(vis, -1, self.cphase_ind[2])

        ang1 = torch.atan2(vis1[:, 1, :], vis1[:, 0, :])
        ang2 = torch.atan2(vis2[:, 1, :], vis2[:, 0, :])
        ang3 = torch.atan2(vis3[:, 1, :], vis3[:, 0, :])
        cphase = (self.cphase_sign[0] * ang1 +
                  self.cphase_sign[1] * ang2 +
                  self.cphase_sign[2] * ang3) * 180 / np.pi

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


def Loss_visamp_diff(sigma, device):
    """Visibility amplitude squared difference loss."""
    sigma = torch.Tensor(sigma).type(torch.float32).to(device=device)

    def func(y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2 / sigma ** 2, 1)
    return func
