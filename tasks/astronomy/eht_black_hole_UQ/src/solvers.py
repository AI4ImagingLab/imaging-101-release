"""
DPI Solver — Real-NVP Normalizing Flow for Posterior Imaging
=============================================================

Implements the Deep Probabilistic Imaging (DPI) method: a Real-NVP
normalizing flow that maps standard Gaussian latent variables to the
posterior distribution over images given interferometric measurements.

Architecture (from Sun & Bouman 2020):
  - Real-NVP with n_flow Flow blocks (default: 16)
  - Each Flow block: ActNorm → AffineCoupling → Reverse → ActNorm → AffineCoupling → Reverse
  - Affine coupling network: 2 hidden layers with LeakyReLU + BatchNorm1d
  - Random permutation between blocks (seeded by block index)
  - Softplus + learnable scale for image positivity

Reference
---------
Sun & Bouman (2020), arXiv:2010.14462 — Deep Probabilistic Imaging
Original code: DPI/DPItorch/generative_model/realnvpfc_model.py
               DPI/DPItorch/DPI_interferometry.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .physics_model import (
    NUFFTForwardModel,
    Loss_angle_diff, Loss_logca_diff2, Loss_logamp_diff,
    Loss_l1, Loss_TSV, Loss_flux, Loss_center, Loss_cross_entropy,
)


# ── Real-NVP Components ────────────────────────────────────────────────────

class ActNorm(nn.Module):
    """
    Activation Normalization layer with data-dependent initialization.

    Learns a per-element location and log-scale, initialized from the
    statistics of the first mini-batch.
    """

    def __init__(self, logdet=True):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1,))
        self.log_scale_inv = nn.Parameter(torch.zeros(1,))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input, inv_init=False):
        with torch.no_grad():
            mean = input.mean().reshape((1,))
            std = input.std().reshape((1,))
            if inv_init:
                self.loc.data.copy_(torch.zeros_like(mean))
                self.log_scale_inv.data.copy_(torch.zeros_like(std))
            else:
                self.loc.data.copy_(-mean)
                self.log_scale_inv.data.copy_(torch.log(std + 1e-6))

    def forward(self, input):
        _, in_dim = input.shape
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        scale_inv = torch.exp(self.log_scale_inv)
        log_abs = -self.log_scale_inv
        logdet = in_dim * torch.sum(log_abs)

        if self.logdet:
            return (1.0 / scale_inv) * (input + self.loc), logdet
        else:
            return (1.0 / scale_inv) * (input + self.loc)

    def reverse(self, output):
        _, in_dim = output.shape
        if self.initialized.item() == 0:
            self.initialize(output, inv_init=True)
            self.initialized.fill_(1)

        scale_inv = torch.exp(self.log_scale_inv)
        log_abs = -self.log_scale_inv
        logdet = -in_dim * torch.sum(log_abs)

        if self.logdet:
            return output * scale_inv - self.loc, logdet
        else:
            return output * scale_inv - self.loc


class ZeroFC(nn.Module):
    """
    Zero-initialized fully connected layer with learnable exponential scale.

    The output is: FC(input) * exp(scale * 3), where FC weights and biases
    start at zero, and scale starts at zero (so initial output is zero).
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(out_dim,))

    def forward(self, input):
        out = self.fc(input)
        out = out * torch.exp(self.scale * 3)
        return out


class AffineCoupling(nn.Module):
    """
    Affine coupling layer for Real-NVP.

    Splits input into two halves (a, b). Applies a neural network to a
    to produce scale (s) and shift (t), then transforms b:
        out_b = (b + t) * exp(tanh(log_s))

    Network architecture: Linear → LeakyReLU → BatchNorm1d → Linear →
    LeakyReLU → BatchNorm1d → ZeroFC

    Parameters
    ----------
    ndim : int
        Total input dimension.
    seqfrac : int
        Compression factor for hidden dimension: hidden = ndim / (2 * seqfrac).
    affine : bool
        If True, apply affine transform; if False, additive only.
    batch_norm : bool
        If True, use BatchNorm1d in the network.
    """

    def __init__(self, ndim, seqfrac=4, affine=True, batch_norm=True):
        super().__init__()
        self.affine = affine
        hidden = int(ndim / (2 * seqfrac))
        in_half = ndim - ndim // 2
        out_dim = 2 * (ndim // 2) if affine else ndim // 2

        if batch_norm:
            self.net = nn.Sequential(
                nn.Linear(in_half, hidden),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.BatchNorm1d(hidden, eps=1e-2, affine=True),
                nn.Linear(hidden, hidden),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.BatchNorm1d(hidden, eps=1e-2, affine=True),
                ZeroFC(hidden, out_dim),
            )
            self.net[0].weight.data.normal_(0, 0.05)
            self.net[0].bias.data.zero_()
            self.net[3].weight.data.normal_(0, 0.05)
            self.net[3].bias.data.zero_()
        else:
            self.net = nn.Sequential(
                nn.Linear(in_half, hidden),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Linear(hidden, hidden),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                ZeroFC(hidden, out_dim),
            )
            self.net[0].weight.data.normal_(0, 0.05)
            self.net[0].bias.data.zero_()
            self.net[2].weight.data.normal_(0, 0.05)
            self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)
        if self.affine:
            log_s0, t = self.net(in_a).chunk(2, 1)
            log_s = torch.tanh(log_s0)
            s = torch.exp(log_s)
            out_b = (in_b + t) * s
            logdet = torch.sum(log_s.view(input.shape[0], -1), 1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None
        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.affine:
            log_s0, t = self.net(out_a).chunk(2, 1)
            log_s = torch.tanh(log_s0)
            s = torch.exp(log_s)
            in_b = out_b / s - t
            logdet = -torch.sum(log_s.view(output.shape[0], -1), 1)
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out
            logdet = None
        return torch.cat([out_a, in_b], 1), logdet


class Flow(nn.Module):
    """
    One flow block: alternating affine coupling with ActNorm.

    Structure: ActNorm → AffineCoupling → Reverse → ActNorm → AffineCoupling → Reverse
    """

    def __init__(self, ndim, affine=True, seqfrac=4, batch_norm=True):
        super().__init__()
        self.actnorm = ActNorm()
        self.actnorm2 = ActNorm()
        self.coupling = AffineCoupling(ndim, seqfrac=seqfrac, affine=affine,
                                        batch_norm=batch_norm)
        self.coupling2 = AffineCoupling(ndim, seqfrac=seqfrac, affine=affine,
                                         batch_norm=batch_norm)
        self.ndim = ndim
        self._reverse_idx = np.arange(ndim - 1, -1, -1)

    def forward(self, input):
        logdet = 0
        out, det1 = self.actnorm(input)
        out, det2 = self.coupling(out)
        out = out[:, self._reverse_idx]
        out, det3 = self.actnorm2(out)
        out, det4 = self.coupling2(out)
        out = out[:, self._reverse_idx]

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        logdet = logdet + det3
        if det4 is not None:
            logdet = logdet + det4
        return out, logdet

    def reverse(self, output):
        logdet = 0
        input = output[:, self._reverse_idx]
        input, det1 = self.coupling2.reverse(input)
        input, det2 = self.actnorm2.reverse(input)
        input = input[:, self._reverse_idx]
        input, det3 = self.coupling.reverse(input)
        input, det4 = self.actnorm.reverse(input)

        if det1 is not None:
            logdet = logdet + det1
        logdet = logdet + det2
        if det3 is not None:
            logdet = logdet + det3
        logdet = logdet + det4
        return input, logdet


def _order_inverse(order):
    """Compute the inverse of a permutation."""
    order_inv = np.empty_like(order)
    for k in range(len(order)):
        order_inv[order[k]] = k
    return order_inv


class RealNVP(nn.Module):
    """
    Real-NVP normalizing flow.

    Stacks n_flow Flow blocks with random permutations between them.
    Permutations are deterministic (seeded by block index) for reproducibility.

    Parameters
    ----------
    ndim : int
        Dimension of the data (npix * npix).
    n_flow : int
        Number of flow blocks.
    affine : bool
        Use affine coupling (True) or additive (False).
    seqfrac : int
        Hidden dimension compression factor.
    permute : str
        Permutation type: 'random', 'reverse', or 'none'.
    batch_norm : bool
        Use BatchNorm in coupling networks.
    """

    def __init__(self, ndim, n_flow, affine=True, seqfrac=4,
                 permute='random', batch_norm=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.orders = []
        for i in range(n_flow):
            self.blocks.append(Flow(ndim, affine=affine, seqfrac=seqfrac,
                                     batch_norm=batch_norm))
            if permute == 'random':
                self.orders.append(
                    np.random.RandomState(seed=i).permutation(ndim))
            elif permute == 'reverse':
                self.orders.append(np.arange(ndim - 1, -1, -1))
            else:
                self.orders.append(np.arange(ndim))

        self.inverse_orders = [_order_inverse(o) for o in self.orders]

    def forward(self, input):
        logdet = 0
        out = input
        for i in range(len(self.blocks)):
            out, det = self.blocks[i](out)
            logdet = logdet + det
            out = out[:, self.orders[i]]
        return out, logdet

    def reverse(self, out):
        logdet = 0
        input = out
        for i in range(len(self.blocks) - 1, -1, -1):
            input = input[:, self.inverse_orders[i]]
            input, det = self.blocks[i].reverse(input)
            logdet = logdet + det
        return input, logdet


class Img_logscale(nn.Module):
    """
    Learnable log-scale factor for image intensity.

    Initialized to log(scale) so that exp(log_scale) = scale.
    """

    def __init__(self, scale=1.0):
        super().__init__()
        log_scale = torch.Tensor(np.log(scale) * np.ones(1))
        self.log_scale = nn.Parameter(log_scale)

    def forward(self):
        return self.log_scale


# ── DPI Solver ──────────────────────────────────────────────────────────────

class DPISolver:
    """
    Deep Probabilistic Imaging solver.

    Trains a Real-NVP normalizing flow to approximate the posterior distribution
    over images given interferometric closure measurements.

    Parameters
    ----------
    npix : int
        Image size (pixels per side). Default: 32.
    n_flow : int
        Number of Real-NVP flow blocks. Default: 16.
    seqfrac : int
        Hidden dimension compression factor. Default: 4.
    n_epoch : int
        Number of training epochs. Default: 10000.
    batch_size : int
        Training mini-batch size. Default: 32.
    lr : float
        Adam learning rate. Default: 1e-4.
    logdet_weight : float
        Entropy term weight. Default: 1.0.
    l1_weight : float
        L1 sparsity prior weight. Default: 1.0.
    tsv_weight : float
        Total squared variation weight. Default: 100.0.
    flux_weight : float
        Flux constraint weight. Default: 1000.0.
    center_weight : float
        Centering constraint weight. Default: 1.0.
    mem_weight : float
        Maximum entropy prior weight. Default: 1024.0.
    grad_clip : float
        Gradient clipping threshold. Default: 0.1.
    data_warmup : bool
        Enable data weight warm-up schedule. Default: True.
    warmup_starting_order : int
        Initial data weight = 10^(-starting_order). Default: 4.
    warmup_decay_rate : int
        Epochs for data weight to increase by one order of magnitude. Default: 2000.
    device : torch.device or None
        Computation device. Auto-detects if None.
    """

    def __init__(self, npix=32, n_flow=16, seqfrac=4, n_epoch=10000,
                 batch_size=32, lr=1e-4, logdet_weight=1.0, l1_weight=1.0,
                 tsv_weight=100.0, flux_weight=1000.0, center_weight=1.0,
                 mem_weight=1024.0, grad_clip=0.1,
                 data_warmup=True, warmup_starting_order=4,
                 warmup_decay_rate=2000, device=None):
        self.npix = npix
        self.n_flow = n_flow
        self.seqfrac = seqfrac
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.logdet_weight = logdet_weight
        self.l1_weight = l1_weight
        self.tsv_weight = tsv_weight
        self.flux_weight = flux_weight
        self.center_weight = center_weight
        self.mem_weight = mem_weight
        self.grad_clip = grad_clip
        self.data_warmup = data_warmup
        self.warmup_starting_order = warmup_starting_order
        self.warmup_decay_rate = warmup_decay_rate

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Set after training
        self.img_generator = None
        self.logscale_factor = None

    def reconstruct(self, obs_data, closure_indices, nufft_params,
                    prior_image, flux_const) -> dict:
        """
        Train the DPI model and return the trained components.

        Parameters
        ----------
        obs_data : dict
            From preprocessing.load_observation(). Must contain 'obs' key.
        closure_indices : dict
            From preprocessing.extract_closure_indices().
        nufft_params : dict
            From preprocessing.compute_nufft_params().
        prior_image : ndarray
            (npix, npix) prior image for MEM regularizer.
        flux_const : float
            Estimated total flux.

        Returns
        -------
        dict with keys:
            'loss_history' : dict of loss component arrays
            'img_generator' : trained RealNVP model
            'logscale_factor' : trained scale parameter
        """
        npix = self.npix
        device = self.device
        obs = obs_data["obs"]

        # Build forward model
        cphase_ind_torch = [torch.tensor(a, dtype=torch.long)
                            for a in closure_indices["cphase_ind_list"]]
        cphase_sign_torch = [torch.tensor(a, dtype=torch.float32)
                             for a in closure_indices["cphase_sign_list"]]
        camp_ind_torch = [torch.tensor(a, dtype=torch.long)
                          for a in closure_indices["camp_ind_list"]]

        forward_model = NUFFTForwardModel(
            npix, nufft_params["ktraj_vis"], nufft_params["pulsefac_vis"],
            cphase_ind_torch, cphase_sign_torch, camp_ind_torch, device
        )

        # Build generative model
        self.img_generator = RealNVP(
            npix * npix, self.n_flow, affine=True, seqfrac=self.seqfrac
        ).to(device)
        self.logscale_factor = Img_logscale(
            scale=flux_const / (0.8 * npix * npix)
        ).to(device)

        # Loss functions
        cphase_data = closure_indices["cphase_data"]
        camp_data = closure_indices["camp_data"]
        logcamp_data = closure_indices["logcamp_data"]

        Loss_cphase_img = Loss_angle_diff(cphase_data['sigmacp'], device)
        Loss_logca_img = Loss_logca_diff2(logcamp_data['sigmaca'], device)
        Loss_logamp_img = Loss_logamp_diff(obs.data['sigma'], device)
        Loss_center_img = Loss_center(device, center=npix / 2 - 0.5, dim=npix)
        Loss_flux_img = Loss_flux(flux_const)

        # Loss weights (faithful to reference defaults)
        camp_weight = 1.0
        cphase_weight = len(cphase_data['cphase']) / len(camp_data['camp'])
        visamp_weight = 0.0
        imgl1_weight = self.l1_weight * npix * npix / flux_const
        imgtsv_weight = self.tsv_weight * npix * npix
        imgflux_weight = self.flux_weight
        imgcenter_weight = self.center_weight * 1e5 / (npix * npix)
        imgcrossentropy_weight = self.mem_weight
        logdet_weight = 2.0 * self.logdet_weight / len(camp_data['camp'])

        # Observed data as tensors
        vis_true_np = obs.data['vis']
        visamp_true = torch.Tensor(np.abs(vis_true_np)).to(device)
        cphase_true = torch.Tensor(np.array(cphase_data['cphase'])).to(device)
        logcamp_true = torch.Tensor(np.array(logcamp_data['camp'])).to(device)
        prior_im = torch.Tensor(prior_image).to(device)

        # Optimizer
        optimizer = optim.Adam(
            list(self.img_generator.parameters()) +
            list(self.logscale_factor.parameters()),
            lr=self.lr
        )

        # Training loop
        loss_history = {
            'total': [], 'cphase': [], 'logca': [], 'visamp': [],
            'logdet': [], 'flux': [], 'tsv': [], 'center': [],
            'mem': [], 'l1': []
        }

        for k in range(self.n_epoch):
            z_sample = torch.randn(self.batch_size, npix * npix).to(device)

            # Generate image samples
            img_samp, logdet = self.img_generator.reverse(z_sample)
            img_samp = img_samp.reshape((-1, npix, npix))

            # Apply scale factor and softplus for positivity
            logscale_factor_value = self.logscale_factor.forward()
            scale_factor = torch.exp(logscale_factor_value)
            img = torch.nn.Softplus()(img_samp) * scale_factor

            # Log-det correction for softplus and scale
            det_softplus = torch.sum(
                img_samp - torch.nn.Softplus()(img_samp), (1, 2))
            det_scale = logscale_factor_value * npix * npix
            logdet = logdet + det_softplus + det_scale

            # Forward model
            vis, visamp, cphase, logcamp = forward_model(img)

            # Compute losses
            loss_cphase = Loss_cphase_img(cphase_true, cphase) if cphase_weight > 0 else 0
            loss_camp = Loss_logca_img(logcamp_true, logcamp) if camp_weight > 0 else 0
            loss_visamp = Loss_logamp_img(visamp_true, visamp) if visamp_weight > 0 else 0

            loss_l1 = Loss_l1(img) if imgl1_weight > 0 else 0
            loss_tsv = Loss_TSV(img) if imgtsv_weight > 0 else 0
            loss_flux = Loss_flux_img(img) if imgflux_weight > 0 else 0
            loss_center = Loss_center_img(img) if imgcenter_weight > 0 else 0
            loss_cross_entropy = Loss_cross_entropy(prior_im, img) if imgcrossentropy_weight > 0 else 0

            loss_data = (camp_weight * loss_camp +
                         cphase_weight * loss_cphase +
                         visamp_weight * loss_visamp)
            loss_prior = (imgcrossentropy_weight * loss_cross_entropy +
                          imgflux_weight * loss_flux +
                          imgtsv_weight * loss_tsv +
                          imgcenter_weight * loss_center +
                          imgl1_weight * loss_l1)

            # Data weight warm-up: gradually increase data fidelity weight
            if self.data_warmup:
                data_weight = min(
                    10 ** (-self.warmup_starting_order + k / self.warmup_decay_rate),
                    1.0)
            else:
                data_weight = 1.0

            loss = (data_weight * torch.mean(loss_data) + torch.mean(loss_prior)
                    - logdet_weight * torch.mean(logdet))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.img_generator.parameters()) +
                list(self.logscale_factor.parameters()),
                self.grad_clip
            )
            optimizer.step()

            # Log losses
            loss_history['total'].append(loss.detach().cpu().item())
            loss_history['cphase'].append(
                torch.mean(loss_cphase).detach().cpu().item() if cphase_weight > 0 else 0)
            loss_history['logca'].append(
                torch.mean(loss_camp).detach().cpu().item() if camp_weight > 0 else 0)
            loss_history['visamp'].append(
                torch.mean(loss_visamp).detach().cpu().item() if visamp_weight > 0 else 0)
            loss_history['logdet'].append(
                -torch.mean(logdet).detach().cpu().item() / (npix * npix))
            loss_history['flux'].append(
                torch.mean(loss_flux).detach().cpu().item() if imgflux_weight > 0 else 0)
            loss_history['tsv'].append(
                torch.mean(loss_tsv).detach().cpu().item() if imgtsv_weight > 0 else 0)
            loss_history['center'].append(
                torch.mean(loss_center).detach().cpu().item() if imgcenter_weight > 0 else 0)
            loss_history['mem'].append(
                torch.mean(loss_cross_entropy).detach().cpu().item() if imgcrossentropy_weight > 0 else 0)
            loss_history['l1'].append(
                torch.mean(loss_l1).detach().cpu().item() if imgl1_weight > 0 else 0)

            if k % 1000 == 0 or k == self.n_epoch - 1:
                print(f"  epoch {k:>5d}/{self.n_epoch}: "
                      f"loss={loss_history['total'][-1]:.4f}  "
                      f"cphase={loss_history['cphase'][-1]:.4f}  "
                      f"camp={loss_history['logca'][-1]:.4f}  "
                      f"logdet={loss_history['logdet'][-1]:.4f}  "
                      f"dw={data_weight:.4f}")

        # Convert to arrays
        for key in loss_history:
            loss_history[key] = np.array(loss_history[key])

        return {
            'loss_history': loss_history,
            'img_generator': self.img_generator,
            'logscale_factor': self.logscale_factor,
        }

    @torch.no_grad()
    def sample(self, n_samples: int = 1000) -> np.ndarray:
        """
        Draw posterior samples from the trained flow.

        Parameters
        ----------
        n_samples : int
            Number of posterior samples to generate.

        Returns
        -------
        (n_samples, npix, npix) ndarray — posterior image samples
        """
        if self.img_generator is None:
            raise RuntimeError("Must call reconstruct() before sample().")

        npix = self.npix
        device = self.device
        self.img_generator.eval()

        logscale_value = self.logscale_factor.forward()
        scale_factor = torch.exp(logscale_value)

        # Generate in batches to avoid OOM
        batch_size = min(n_samples, 128)
        samples = []
        remaining = n_samples

        while remaining > 0:
            bs = min(batch_size, remaining)
            z = torch.randn(bs, npix * npix).to(device)
            img_samp, _ = self.img_generator.reverse(z)
            img_samp = img_samp.reshape(-1, npix, npix)
            img = torch.nn.Softplus()(img_samp) * scale_factor
            samples.append(img.cpu().numpy())
            remaining -= bs

        self.img_generator.train()
        return np.concatenate(samples, axis=0)

    def posterior_statistics(self, n_samples: int = 1000) -> dict:
        """
        Compute posterior statistics from samples.

        Parameters
        ----------
        n_samples : int
            Number of posterior samples.

        Returns
        -------
        dict with keys:
            'mean'    : (npix, npix) ndarray — posterior mean image
            'std'     : (npix, npix) ndarray — posterior standard deviation
            'samples' : (n_samples, npix, npix) ndarray — all samples
        """
        samples = self.sample(n_samples)
        return {
            'mean': np.mean(samples, axis=0),
            'std': np.std(samples, axis=0),
            'samples': samples,
        }
