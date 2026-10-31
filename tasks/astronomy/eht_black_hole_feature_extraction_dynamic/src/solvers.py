"""
α-DPI Solver — Real-NVP Normalizing Flow for Geometric Feature Extraction
===========================================================================

Implements α-DPI (α-deep Probabilistic Inference): a two-step algorithm
that uses α-divergence variational inference + importance sampling to
infer posterior distributions over geometric black hole parameters.

Architecture:
  - Real-NVP normalizing flow maps Gaussian latent → unconstrained params
  - Sigmoid maps to [0,1] unit params → geometric model → image → closure
  - α-divergence training loss (Eq. 1 in Sun et al. 2022)
  - Importance sampling reweighting for accurate posteriors (Step 2)
  - ELBO computation for model selection (Eq. 4)

Reference
---------
Sun et al. (2022), ApJ 932:99 — α-DPI
Original code: DPI/DPItorch/DPIx_interferometry.py
               DPI/DPItorch/generative_model/realnvpfc_model.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import softmax

from .physics_model import (
    NUFFTForwardModel,
    SimpleCrescentParam2Img,
    SimpleCrescentNuisanceParam2Img,
    SimpleCrescentNuisanceFloorParam2Img,
    Loss_angle_diff, Loss_logca_diff2, Loss_visamp_diff,
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
    start at zero, and scale starts at zero.
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

    Parameters
    ----------
    ndim : int
        Total input dimension.
    seqfrac : float
        Compression factor for hidden dimension: hidden = max(2, ndim / (2 * seqfrac)).
    affine : bool
        If True, apply affine transform; if False, additive only.
    batch_norm : bool
        If True, use BatchNorm1d in the network.
    """

    def __init__(self, ndim, seqfrac=1 / 16, affine=True, batch_norm=True):
        super().__init__()
        self.affine = affine
        hidden = max(2, int(ndim / (2 * seqfrac)))
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

    def __init__(self, ndim, affine=True, seqfrac=1 / 16, batch_norm=True):
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

    Parameters
    ----------
    ndim : int
        Dimension of the parameter space.
    n_flow : int
        Number of flow blocks.
    affine : bool
        Use affine coupling (True) or additive (False).
    seqfrac : float
        Hidden dimension compression factor.
    permute : str
        Permutation type: 'random', 'reverse', or 'none'.
    batch_norm : bool
        Use BatchNorm in coupling networks.
    """

    def __init__(self, ndim, n_flow, affine=True, seqfrac=1 / 16,
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


# ── α-DPI Solver ──────────────────────────────────────────────────────────

class AlphaDPISolver:
    """
    α-DPI solver for geometric feature extraction.

    Trains a Real-NVP normalizing flow to approximate the posterior
    distribution over geometric parameters given closure measurements,
    using α-divergence variational inference.

    Parameters
    ----------
    npix : int
        Image size for rendering geometric model.
    fov_uas : float
        Field of view in microarcseconds.
    n_flow : int
        Number of Real-NVP flow blocks.
    seqfrac : float
        Hidden dimension compression factor.
    n_epoch : int
        Number of training epochs.
    batch_size : int
        Training mini-batch size.
    lr : float
        Adam learning rate.
    logdet_weight : float
        Entropy term weight.
    grad_clip : float
        Gradient clipping threshold.
    alpha : float
        α-divergence parameter. α=1.0 is KL divergence.
    beta : float
        Alternative parameterization: α = 1 - β * scale_factor. Default 0.0 (KL divergence).
    start_order : float
        Data warmup starting order (10^-start_order).
    decay_rate : float
        Data warmup decay rate in epochs.
    geometric_model : str
        Model type: 'simple_crescent', 'simple_crescent_nuisance', or 'simple_crescent_floor_nuisance'.
    n_gaussian : int
        Number of nuisance Gaussians (for nuisance model).
    r_range : list
        [min, max] radius in microarcseconds.
    width_range : list
        [min, max] width in microarcseconds.
    device : torch.device or None
        Computation device.
    """

    def __init__(self, npix=64, fov_uas=120.0, n_flow=16, seqfrac=1 / 16,
                 n_epoch=10000, batch_size=2048, lr=1e-4,
                 logdet_weight=1.0, grad_clip=1e-4,
                 alpha=1.0, beta=0.0, start_order=4, decay_rate=2000,
                 geometric_model='simple_crescent_floor_nuisance', n_gaussian=2,
                 r_range=None, width_range=None,
                 shift_range=None, sigma_range=None,
                 floor_range=None, asym_range=None,
                 crescent_flux_range=None, gaussian_scale_range=None,
                 device=None):
        self.npix = npix
        self.fov_uas = fov_uas
        self.n_flow = n_flow
        self.seqfrac = seqfrac
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.logdet_weight = logdet_weight
        self.grad_clip = grad_clip
        self.alpha = alpha
        self.beta = beta
        self.start_order = start_order
        self.decay_rate = decay_rate
        self.geometric_model = geometric_model
        self.n_gaussian = n_gaussian
        self.r_range = r_range or [10.0, 40.0]
        self.width_range = width_range or [1.0, 40.0]
        self.shift_range = shift_range
        self.sigma_range = sigma_range
        self.floor_range = floor_range
        self.asym_range = asym_range
        self.crescent_flux_range = crescent_flux_range
        self.gaussian_scale_range = gaussian_scale_range

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.params_generator = None
        self.img_converter = None

    def _build_geometric_model(self):
        """Build the geometric image model."""
        # Collect optional range kwargs for floor model
        range_kwargs = {}
        if self.shift_range is not None:
            range_kwargs['shift_range'] = self.shift_range
        if self.sigma_range is not None:
            range_kwargs['sigma_range'] = self.sigma_range
        if self.floor_range is not None:
            range_kwargs['floor_range'] = self.floor_range
        if self.asym_range is not None:
            range_kwargs['asym_range'] = self.asym_range
        if self.crescent_flux_range is not None:
            range_kwargs['crescent_flux_range'] = self.crescent_flux_range
        if self.gaussian_scale_range is not None:
            range_kwargs['gaussian_scale_range'] = self.gaussian_scale_range

        if self.geometric_model == 'simple_crescent':
            self.img_converter = SimpleCrescentParam2Img(
                self.npix, fov=self.fov_uas,
                r_range=self.r_range, width_range=self.width_range
            ).to(device=self.device)
        elif self.geometric_model == 'simple_crescent_nuisance':
            self.img_converter = SimpleCrescentNuisanceParam2Img(
                self.npix, n_gaussian=self.n_gaussian, fov=self.fov_uas,
                r_range=self.r_range, width_range=self.width_range
            ).to(device=self.device)
        elif self.geometric_model == 'simple_crescent_floor_nuisance':
            self.img_converter = SimpleCrescentNuisanceFloorParam2Img(
                self.npix, n_gaussian=self.n_gaussian, fov=self.fov_uas,
                r_range=self.r_range, width_range=self.width_range,
                **range_kwargs
            ).to(device=self.device)
        else:
            raise ValueError(f"Unknown geometric model: {self.geometric_model}")
        return self.img_converter.nparams

    def reconstruct(self, obs_data, closure_indices, nufft_params,
                    flux_const) -> dict:
        """
        Train the α-DPI model.

        Parameters
        ----------
        obs_data : dict
            From preprocessing.load_observation().
        closure_indices : dict
            From preprocessing.extract_closure_indices().
        nufft_params : dict
            From preprocessing.compute_nufft_params().
        flux_const : float
            Estimated total flux.

        Returns
        -------
        dict with keys:
            'loss_history'      : dict of loss component arrays
            'params_generator'  : trained RealNVP model
            'img_converter'     : geometric model
        """
        device = self.device

        # Build geometric model
        nparams = self._build_geometric_model()

        # Build forward model
        cphase_ind_torch = [torch.tensor(a, dtype=torch.long)
                            for a in closure_indices["cphase_ind_list"]]
        cphase_sign_torch = [torch.tensor(a, dtype=torch.float32)
                             for a in closure_indices["cphase_sign_list"]]
        camp_ind_torch = [torch.tensor(a, dtype=torch.long)
                          for a in closure_indices["camp_ind_list"]]

        forward_model = NUFFTForwardModel(
            self.npix, nufft_params["ktraj_vis"], nufft_params["pulsefac_vis"],
            cphase_ind_torch, cphase_sign_torch, camp_ind_torch, device
        )

        # Build normalizing flow
        self.params_generator = RealNVP(
            nparams, self.n_flow, affine=True, seqfrac=self.seqfrac,
            permute='random', batch_norm=True
        ).to(device)

        # Loss functions
        cphase_data = closure_indices["cphase_data"]
        camp_data = closure_indices["camp_data"]
        logcamp_data = closure_indices["logcamp_data"]

        Loss_cphase_img = Loss_angle_diff(cphase_data['sigmacp'], device)
        Loss_logca_img = Loss_logca_diff2(logcamp_data['sigmaca'], device)

        # Loss weights
        camp_weight = 1.0
        cphase_weight = len(cphase_data['cphase']) / len(camp_data['camp'])
        logdet_weight = 2.0 * self.logdet_weight / len(camp_data['camp'])
        scale_factor = 1.0 / len(camp_data['camp'])

        # α-divergence parameter
        if self.beta == 0:
            alpha_divergence = self.alpha
        else:
            alpha_divergence = 1 - self.beta * scale_factor

        # Observed data as tensors
        cphase_true = torch.Tensor(np.array(cphase_data['cphase'])).to(device)
        logcamp_true = torch.Tensor(np.array(logcamp_data['camp'])).to(device)

        # Optimizer
        optimizer = optim.Adam(self.params_generator.parameters(), lr=self.lr)

        # Training loop
        loss_history = {
            'total': [], 'cphase': [], 'logca': [], 'logdet': [],
        }
        loss_best = 1e5
        best_state = None
        n_smooth = 10

        for k in range(self.n_epoch):
            data_weight = min(10 ** (-self.start_order + k / self.decay_rate), 1.0)

            z_sample = torch.randn((self.batch_size, nparams)).to(device=device)

            # Generate parameter samples via flow
            params_samp, logdet = self.params_generator.reverse(z_sample)
            params = torch.sigmoid(params_samp)

            # Generate images via geometric model
            img = self.img_converter.forward(params)

            # Log-det correction for sigmoid transform
            det_sigmoid = torch.sum(
                -params_samp - 2 * torch.nn.Softplus()(-params_samp), -1)
            logdet = logdet + det_sigmoid

            # Forward model: image → closure quantities
            vis, visamp, cphase, logcamp = forward_model(img)

            # Compute data fidelity losses
            loss_cphase = Loss_cphase_img(cphase_true, cphase) if cphase_weight > 0 else 0
            loss_camp = Loss_logca_img(logcamp_true, logcamp) if camp_weight > 0 else 0
            loss_data = camp_weight * loss_camp + cphase_weight * loss_cphase
            loss_data = 0.5 * loss_data / scale_factor

            # Log probability under base distribution
            logprob = -logdet - 0.5 * torch.sum(z_sample ** 2, 1)

            # Combined loss with data warmup
            loss = data_weight * loss_data + logprob

            # α-divergence optimization
            if alpha_divergence == 1:
                loss_train = torch.mean(scale_factor * loss)
            else:
                if self.beta == 0:
                    rej_weights = nn.Softmax(dim=0)(
                        -(1 - alpha_divergence) * loss).detach()
                else:
                    rej_weights = nn.Softmax(dim=0)(
                        -self.beta * scale_factor * loss).detach()
                loss_train = torch.sum(rej_weights * scale_factor * loss)

            # Compute ELBO-like objective for logging
            loss_orig = loss_data + logprob
            if alpha_divergence == 1:
                elbo_loss = torch.mean(scale_factor * loss_orig)
            else:
                if self.beta == 0:
                    elbo_loss = scale_factor * torch.log(
                        torch.mean(torch.exp(-(1 - alpha_divergence) * loss_orig))
                    ) / (alpha_divergence - 1)
                else:
                    elbo_loss = scale_factor * torch.log(
                        torch.mean(torch.exp(-self.beta * scale_factor * loss_orig))
                    ) / (alpha_divergence - 1)

            optimizer.zero_grad()
            loss_train.backward()
            nn.utils.clip_grad_norm_(self.params_generator.parameters(),
                                      self.grad_clip)
            optimizer.step()

            # Log losses
            loss_history['total'].append(elbo_loss.detach().cpu().item())
            loss_history['cphase'].append(
                torch.mean(loss_cphase).detach().cpu().item() if cphase_weight > 0 else 0)
            loss_history['logca'].append(
                torch.mean(loss_camp).detach().cpu().item() if camp_weight > 0 else 0)
            loss_history['logdet'].append(
                -torch.mean(logdet).detach().cpu().item() / nparams)

            # Best model checkpointing
            if k > n_smooth + 1:
                loss_now = np.mean(loss_history['total'][-n_smooth:])
                if loss_now <= loss_best:
                    loss_best = loss_now
                    best_state = {
                        key: val.cpu().clone()
                        for key, val in self.params_generator.state_dict().items()
                    }

            if k % 1000 == 0 or k == self.n_epoch - 1:
                print(f"  epoch {k:>5d}/{self.n_epoch}: "
                      f"loss={loss_history['total'][-1]:.4f}  "
                      f"cphase={loss_history['cphase'][-1]:.4f}  "
                      f"camp={loss_history['logca'][-1]:.4f}  "
                      f"logdet={loss_history['logdet'][-1]:.4f}")

        # Restore best model
        if best_state is not None:
            self.params_generator.load_state_dict(best_state)

        # Convert to arrays
        for key in loss_history:
            loss_history[key] = np.array(loss_history[key])

        return {
            'loss_history': loss_history,
            'params_generator': self.params_generator,
            'img_converter': self.img_converter,
        }

    @torch.no_grad()
    def sample(self, n_samples: int = 10000) -> dict:
        """
        Draw posterior parameter samples from the trained flow.

        Parameters
        ----------
        n_samples : int
            Number of posterior samples.

        Returns
        -------
        dict with keys:
            'params_unit'  : (n_samples, nparams) ndarray — [0,1] params
            'params_samp'  : (n_samples, nparams) ndarray — unconstrained params
            'z_samples'    : (n_samples, nparams) ndarray — latent samples
        """
        if self.params_generator is None:
            raise RuntimeError("Must call reconstruct() before sample().")

        nparams = self.img_converter.nparams
        device = self.device
        self.params_generator.eval()

        batch_size = min(n_samples, 2048)
        all_params_unit = []
        all_params_samp = []
        all_z = []
        remaining = n_samples

        while remaining > 0:
            bs = min(batch_size, remaining)
            z = torch.randn(bs, nparams).to(device)
            params_samp, logdet = self.params_generator.reverse(z)
            params = torch.sigmoid(params_samp)

            all_params_unit.append(params.cpu().numpy())
            all_params_samp.append(params_samp.cpu().numpy())
            all_z.append(z.cpu().numpy())
            remaining -= bs

        self.params_generator.train()
        return {
            'params_unit': np.concatenate(all_params_unit, axis=0),
            'params_samp': np.concatenate(all_params_samp, axis=0),
            'z_samples': np.concatenate(all_z, axis=0),
        }

    def extract_physical_params(self, params_unit: np.ndarray) -> np.ndarray:
        """
        Convert unit-interval parameters to physical units.

        Parameters
        ----------
        params_unit : (N, nparams) ndarray — parameters in [0, 1]

        Returns
        -------
        (N, n_physical) ndarray with columns:
            For simple_crescent: [diameter_uas, width_uas, asymmetry, PA_deg]
            For nuisance: + [x, y, scale, sigma_x, sigma_y, rho] * n_gaussian
            For floor_nuisance: + [x, y, scale, sigma_x, sigma_y, theta] * n_gaussian + [floor, crescent_flux]
        """
        fov = self.fov_uas
        r_range = self.r_range
        width_range = self.width_range
        p = params_unit

        diameter = 2 * (r_range[0] + p[:, 0] * (r_range[1] - r_range[0]))
        width = width_range[0] + p[:, 1] * (width_range[1] - width_range[0])
        pa_deg = 181 / 180 * 180 * (2.0 * p[:, 3] - 1.0)

        if self.geometric_model == 'simple_crescent_floor_nuisance':
            model = self.img_converter
            asymmetry = model.asym_range[0] + p[:, 2] * (model.asym_range[1] - model.asym_range[0])
            cols = [diameter, width, asymmetry, pa_deg]

            for k in range(self.n_gaussian):
                x_shift = model.shift_range[0] + p[:, 4 + k * 6] * (model.shift_range[1] - model.shift_range[0])
                y_shift = model.shift_range[0] + p[:, 5 + k * 6] * (model.shift_range[1] - model.shift_range[0])
                scale = model.gaussian_scale_range[0] + p[:, 6 + k * 6] * (model.gaussian_scale_range[1] - model.gaussian_scale_range[0])
                sigma_x = model.sigma_range[0] + p[:, 7 + k * 6] * (model.sigma_range[1] - model.sigma_range[0])
                sigma_y = model.sigma_range[0] + p[:, 8 + k * 6] * (model.sigma_range[1] - model.sigma_range[0])
                theta_deg = 181 / 180 * 90 * p[:, 9 + k * 6]
                cols.extend([x_shift, y_shift, scale, sigma_x, sigma_y, theta_deg])

            floor = model.floor_range[0] + p[:, 4 + self.n_gaussian * 6] * (model.floor_range[1] - model.floor_range[0])
            crescent_flux = model.crescent_flux_range[0] + p[:, 5 + self.n_gaussian * 6] * (model.crescent_flux_range[1] - model.crescent_flux_range[0])
            cols.extend([floor, crescent_flux])

        elif self.geometric_model == 'simple_crescent_nuisance':
            asymmetry = p[:, 2]
            cols = [diameter, width, asymmetry, pa_deg]
            for k in range(self.n_gaussian):
                x_shift = (2 * p[:, 4 + k * 6] - 1) * 0.5 * fov
                y_shift = (2 * p[:, 5 + k * 6] - 1) * 0.5 * fov
                scale = p[:, 6 + k * 6]
                sigma_x = p[:, 7 + k * 6] * 0.5 * fov
                sigma_y = p[:, 8 + k * 6] * 0.5 * fov
                rho = 2 * 0.99 * (p[:, 9 + k * 6] - 0.5)
                cols.extend([x_shift, y_shift, scale, sigma_x, sigma_y, rho])
        else:
            asymmetry = p[:, 2]
            cols = [diameter, width, asymmetry, pa_deg]

        return np.column_stack(cols)

    @torch.no_grad()
    def importance_resample(self, obs_data, closure_indices, nufft_params,
                            n_samples: int = 10000) -> dict:
        """
        Importance sampling reweighting (Step 2 of α-DPI).

        Generates samples from the trained flow and reweights them by
        p(y|x)p(x) / q_θ(x) to correct the approximate posterior.

        Parameters
        ----------
        obs_data : dict
            From preprocessing.load_observation().
        closure_indices : dict
            From preprocessing.extract_closure_indices().
        nufft_params : dict
            From preprocessing.compute_nufft_params().
        n_samples : int
            Number of samples for importance sampling.

        Returns
        -------
        dict with keys:
            'params_physical'      : (n_samples, n_phys) ndarray
            'importance_weights'   : (n_samples,) ndarray
            'images'               : (n_samples, npix, npix) ndarray
            'weighted_mean_image'  : (npix, npix) ndarray
        """
        if self.params_generator is None:
            raise RuntimeError("Must call reconstruct() before importance_resample().")

        device = self.device
        nparams = self.img_converter.nparams

        # Build forward model
        cphase_ind_torch = [torch.tensor(a, dtype=torch.long)
                            for a in closure_indices["cphase_ind_list"]]
        cphase_sign_torch = [torch.tensor(a, dtype=torch.float32)
                             for a in closure_indices["cphase_sign_list"]]
        camp_ind_torch = [torch.tensor(a, dtype=torch.long)
                          for a in closure_indices["camp_ind_list"]]

        forward_model = NUFFTForwardModel(
            self.npix, nufft_params["ktraj_vis"], nufft_params["pulsefac_vis"],
            cphase_ind_torch, cphase_sign_torch, camp_ind_torch, device
        )

        # Loss functions
        cphase_data = closure_indices["cphase_data"]
        logcamp_data = closure_indices["logcamp_data"]
        camp_data = closure_indices["camp_data"]

        Loss_cphase_img = Loss_angle_diff(cphase_data['sigmacp'], device)
        Loss_logca_img = Loss_logca_diff2(logcamp_data['sigmaca'], device)

        camp_weight = 1.0
        cphase_weight = len(cphase_data['cphase']) / len(camp_data['camp'])

        cphase_true = torch.Tensor(np.array(cphase_data['cphase'])).to(device)
        logcamp_true = torch.Tensor(np.array(logcamp_data['camp'])).to(device)

        self.params_generator.eval()

        all_params_unit = []
        all_images = []
        all_neg_log_weights = []

        batch_size = min(n_samples, 2048)
        remaining = n_samples

        while remaining > 0:
            bs = min(batch_size, remaining)
            z = torch.randn(bs, nparams).to(device)

            params_samp, logdet = self.params_generator.reverse(z)
            params = torch.sigmoid(params_samp)

            img = self.img_converter.forward(params)

            det_sigmoid = torch.sum(
                -params_samp - 2 * torch.nn.Softplus()(-params_samp), -1)
            logdet = logdet + det_sigmoid

            vis, visamp, cphase, logcamp = forward_model(img)

            loss_cphase = Loss_cphase_img(cphase_true, cphase)
            loss_camp = Loss_logca_img(logcamp_true, logcamp)
            loss_data = camp_weight * loss_camp + cphase_weight * loss_cphase

            logprob = -logdet - 0.5 * torch.sum(z ** 2, 1)
            loss_orig = 0.5 * loss_data + logprob

            all_params_unit.append(params.cpu().numpy())
            all_images.append(img.cpu().numpy())
            all_neg_log_weights.append(loss_orig.cpu().numpy())
            remaining -= bs

        self.params_generator.train()

        params_unit = np.concatenate(all_params_unit, axis=0)
        images = np.concatenate(all_images, axis=0)
        neg_log_weights = np.concatenate(all_neg_log_weights, axis=0)

        # Importance weights via softmax
        importance_weights = softmax(-neg_log_weights)

        # Weighted mean image
        weighted_mean_image = np.sum(
            importance_weights.reshape(-1, 1, 1) * images, axis=0)

        # Physical parameters
        params_physical = self.extract_physical_params(params_unit)

        return {
            'params_physical': params_physical,
            'importance_weights': importance_weights,
            'images': images,
            'weighted_mean_image': weighted_mean_image,
            'params_unit': params_unit,
        }

    @torch.no_grad()
    def compute_elbo(self, obs_data, closure_indices, nufft_params,
                     n_samples: int = 10000) -> float:
        """
        Compute Evidence Lower Bound (ELBO) for model selection.

        ELBO = E_q[log p(y|x)] - KL(q||p) ≈ -mean(loss_data + logprob)

        Parameters
        ----------
        obs_data : dict
        closure_indices : dict
        nufft_params : dict
        n_samples : int

        Returns
        -------
        float — estimated ELBO (higher is better)
        """
        if self.params_generator is None:
            raise RuntimeError("Must call reconstruct() before compute_elbo().")

        device = self.device
        nparams = self.img_converter.nparams

        cphase_ind_torch = [torch.tensor(a, dtype=torch.long)
                            for a in closure_indices["cphase_ind_list"]]
        cphase_sign_torch = [torch.tensor(a, dtype=torch.float32)
                             for a in closure_indices["cphase_sign_list"]]
        camp_ind_torch = [torch.tensor(a, dtype=torch.long)
                          for a in closure_indices["camp_ind_list"]]

        forward_model = NUFFTForwardModel(
            self.npix, nufft_params["ktraj_vis"], nufft_params["pulsefac_vis"],
            cphase_ind_torch, cphase_sign_torch, camp_ind_torch, device
        )

        cphase_data = closure_indices["cphase_data"]
        logcamp_data = closure_indices["logcamp_data"]
        camp_data = closure_indices["camp_data"]

        Loss_cphase_img = Loss_angle_diff(cphase_data['sigmacp'], device)
        Loss_logca_img = Loss_logca_diff2(logcamp_data['sigmaca'], device)

        camp_weight = 1.0
        cphase_weight = len(cphase_data['cphase']) / len(camp_data['camp'])

        cphase_true = torch.Tensor(np.array(cphase_data['cphase'])).to(device)
        logcamp_true = torch.Tensor(np.array(logcamp_data['camp'])).to(device)

        self.params_generator.eval()

        all_losses = []
        batch_size = min(n_samples, 2048)
        remaining = n_samples

        while remaining > 0:
            bs = min(batch_size, remaining)
            z = torch.randn(bs, nparams).to(device)

            params_samp, logdet = self.params_generator.reverse(z)
            params = torch.sigmoid(params_samp)
            img = self.img_converter.forward(params)

            det_sigmoid = torch.sum(
                -params_samp - 2 * torch.nn.Softplus()(-params_samp), -1)
            logdet = logdet + det_sigmoid

            vis, visamp, cphase, logcamp = forward_model(img)

            loss_cphase = Loss_cphase_img(cphase_true, cphase)
            loss_camp = Loss_logca_img(logcamp_true, logcamp)
            loss_data = camp_weight * loss_camp + cphase_weight * loss_cphase

            logprob = -logdet - 0.5 * torch.sum(z ** 2, 1)
            loss_orig = 0.5 * loss_data + logprob

            all_losses.append(loss_orig.cpu().numpy())
            remaining -= bs

        self.params_generator.train()

        all_losses = np.concatenate(all_losses, axis=0)
        elbo = -float(np.mean(all_losses))
        return elbo
