"""
FPM-INR Solver and Network Architecture
========================================

Implicit Neural Representation (INR) model and training/inference pipeline
for Fourier Ptychographic Microscopy reconstruction.

Architecture:
    FullModel
    ├── G_Tensor3D (amplitude)  →  2D spatial features × 1D z-features → MLP → scalar
    └── G_Tensor3D (phase)      →  2D spatial features × 1D z-features → MLP → scalar

Training uses Adam with StepLR decay, smooth L1 loss, mixed-precision (bfloat16),
and alternating z-plane sampling.

Based on the original implementation by Haowen Zhou and Brandon Y. Feng.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# INR Network Components
# ──────────────────────────────────────────────────────────────────────────────


class G_Renderer(nn.Module):
    """MLP that maps feature vectors to scalar values."""

    def __init__(
        self, in_dim=32, hidden_dim=32, num_layers=2, out_dim=1, use_layernorm=False
    ):
        super().__init__()
        act_fn = nn.ReLU()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
        layers.append(act_fn)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
            layers.append(act_fn)

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class G_FeatureTensor(nn.Module):
    """2D learnable feature grid with bilinear interpolation."""

    def __init__(self, x_dim, y_dim, num_feats=32, ds_factor=1):
        super().__init__()
        self.x_dim, self.y_dim = x_dim, y_dim
        x_mode, y_mode = x_dim // ds_factor, y_dim // ds_factor
        self.num_feats = num_feats

        self.data = nn.Parameter(
            2e-4 * torch.rand((x_mode, y_mode, num_feats)) - 1e-4, requires_grad=True
        )

        half_dx, half_dy = 0.5 / x_dim, 0.5 / y_dim
        xs = torch.linspace(half_dx, 1 - half_dx, x_dim)
        ys = torch.linspace(half_dx, 1 - half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()

        xs = xy * torch.tensor([x_mode, y_mode], device=xs.device).float()
        indices = xs.long()
        self.lerp_weights = nn.Parameter(xs - indices.float(), requires_grad=False)

        self.x0 = nn.Parameter(
            indices[:, 0].clamp(min=0, max=x_mode - 1), requires_grad=False
        )
        self.y0 = nn.Parameter(
            indices[:, 1].clamp(min=0, max=y_mode - 1), requires_grad=False
        )
        self.x1 = nn.Parameter((self.x0 + 1).clamp(max=x_mode - 1), requires_grad=False)
        self.y1 = nn.Parameter((self.y0 + 1).clamp(max=y_mode - 1), requires_grad=False)

    def sample(self):
        return (
            self.data[self.y0, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y0, self.x1]
            * self.lerp_weights[:, 0:1]
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y1, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * self.lerp_weights[:, 1:2]
            + self.data[self.y1, self.x1]
            * self.lerp_weights[:, 0:1]
            * self.lerp_weights[:, 1:2]
        )

    def forward(self):
        return self.sample()


class G_Tensor(G_FeatureTensor):
    """2D feature grid + renderer wrapper."""

    def __init__(self, im_size, num_feats=32, ds_factor=1):
        super().__init__(im_size, im_size, num_feats=num_feats, ds_factor=ds_factor)
        self.renderer = G_Renderer(in_dim=num_feats)

    def forward(self):
        feats = self.sample()
        return self.renderer(feats)


class G_Tensor3D(nn.Module):
    """
    3D implicit representation: 2D spatial features x 1D z-features.

    Spatial features are stored on a learnable 2D grid and sampled via bilinear
    interpolation. Depth features are stored as a 1D array and linearly interpolated.
    The element-wise product of spatial and depth features is passed through an MLP
    renderer to produce a scalar field at each (x, y, z) location.
    """

    def __init__(
        self, x_mode, y_mode, z_dim, z_min, z_max, num_feats=32, use_layernorm=False
    ):
        super().__init__()
        self.x_mode, self.y_mode, self.num_feats = x_mode, y_mode, num_feats
        self.data = nn.Parameter(
            2e-4 * torch.randn((self.x_mode, self.y_mode, self.num_feats)),
            requires_grad=True,
        )
        self.renderer = G_Renderer(in_dim=self.num_feats, use_layernorm=use_layernorm)
        self.x0 = None

        self.z_mode = z_dim
        self.z_data = nn.Parameter(
            torch.randn((self.z_mode, self.num_feats)), requires_grad=True
        )
        self.z_min = z_min
        self.z_max = z_max
        self.z_dim = z_dim

    def create_coords(self, x_dim, y_dim, x_max, y_max):
        half_dx, half_dy = 0.5 / x_dim, 0.5 / y_dim
        xs = torch.linspace(half_dx, 1 - half_dx, x_dim)
        ys = torch.linspace(half_dx, 1 - half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()
        xs = xy * torch.tensor([x_max, y_max], device=xs.device).float()
        indices = xs.long()
        self.x_dim, self.y_dim = x_dim, y_dim
        self.xy_coords = nn.Parameter(
            xy[None],
            requires_grad=False,
        )

        if self.x0 is not None:
            device = self.x0.device
            self.x0.data = (indices[:, 0].clamp(min=0, max=x_max - 1)).to(device)
            self.y0.data = indices[:, 1].clamp(min=0, max=y_max - 1).to(device)
            self.x1.data = (self.x0 + 1).clamp(max=x_max - 1).to(device)
            self.y1.data = (self.y0 + 1).clamp(max=y_max - 1).to(device)
            self.lerp_weights.data = (xs - indices.float()).to(device)
        else:
            self.x0 = nn.Parameter(
                indices[:, 0].clamp(min=0, max=x_max - 1),
                requires_grad=False,
            )
            self.y0 = nn.Parameter(
                indices[:, 1].clamp(min=0, max=y_max - 1),
                requires_grad=False,
            )
            self.x1 = nn.Parameter(
                (self.x0 + 1).clamp(max=x_max - 1), requires_grad=False
            )
            self.y1 = nn.Parameter(
                (self.y0 + 1).clamp(max=y_max - 1), requires_grad=False
            )
            self.lerp_weights = nn.Parameter(xs - indices.float(), requires_grad=False)

    def normalize_z(self, z):
        return (self.z_dim - 1) * (z - self.z_min) / (self.z_max - self.z_min)

    def sample(self, z):
        z = self.normalize_z(z)
        z0 = z.long().clamp(min=0, max=self.z_dim - 1)
        z1 = (z0 + 1).clamp(max=self.z_dim - 1)
        zlerp_weights = (z - z.long().float())[:, None]

        xy_feat = (
            self.data[self.y0, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y0, self.x1]
            * self.lerp_weights[:, 0:1]
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y1, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * self.lerp_weights[:, 1:2]
            + self.data[self.y1, self.x1]
            * self.lerp_weights[:, 0:1]
            * self.lerp_weights[:, 1:2]
        )
        z_feat = (
            self.z_data[z0] * (1.0 - zlerp_weights) + self.z_data[z1] * zlerp_weights
        )
        z_feat = z_feat[:, None].repeat(1, xy_feat.shape[0], 1)

        feat = xy_feat[None].repeat(z.shape[0], 1, 1) * z_feat

        return feat

    def forward(self, z):
        feat = self.sample(z)

        out = self.renderer(feat)
        b = z.shape[0]
        w, h = self.x_dim, self.y_dim
        out = out.view(b, 1, w, h)

        return out


class FullModel(nn.Module):
    """
    Complete FPM-INR model: dual G_Tensor3D for amplitude and phase.

    Parameters
    ----------
    w, h : int
        Output image dimensions (upsampled).
    num_feats : int
        Feature dimension for INR.
    x_mode, y_mode : int
        Spatial feature grid resolution.
    z_min, z_max : float
        Depth range in micrometers.
    ds_factor : int
        Downsampling factor for coarse-to-fine (1 = full resolution).
    use_layernorm : bool
        Whether to use LayerNorm in the MLP renderer.
    """

    def __init__(
        self, w, h, num_feats, x_mode, y_mode, z_min, z_max, ds_factor, use_layernorm
    ):
        super().__init__()
        self.img_real = G_Tensor3D(
            x_mode=x_mode,
            y_mode=y_mode,
            z_dim=5,
            z_min=z_min,
            z_max=z_max,
            num_feats=num_feats,
            use_layernorm=use_layernorm,
        )
        self.img_imag = G_Tensor3D(
            x_mode=x_mode,
            y_mode=y_mode,
            z_dim=5,
            z_min=z_min,
            z_max=z_max,
            num_feats=num_feats,
            use_layernorm=use_layernorm,
        )
        self.w, self.h = w, h
        self.init_scale_grids(ds_factor=ds_factor)

    def init_scale_grids(self, ds_factor):
        self.img_real.create_coords(
            x_dim=self.w // ds_factor,
            y_dim=self.h // ds_factor,
            x_max=self.img_real.x_mode,
            y_max=self.img_real.y_mode,
        )
        self.img_imag.create_coords(
            x_dim=self.w // ds_factor,
            y_dim=self.h // ds_factor,
            x_max=self.img_imag.x_mode,
            y_max=self.img_imag.y_mode,
        )
        self.ds_factor = ds_factor
        self.us_module = nn.Upsample(scale_factor=ds_factor, mode="bilinear")

    def forward(self, dz):
        img_real = self.img_real(dz)
        img_imag = self.img_imag(dz)
        img_real = self.us_module(img_real).squeeze(1)
        img_imag = self.us_module(img_imag).squeeze(1)

        return img_real, img_imag


def save_model_with_required_grad(model, save_path):
    """Save only parameters with requires_grad=True (compact checkpoint)."""
    tensors_to_save = []
    for param_name, param_tensor in model.named_parameters():
        if param_tensor.requires_grad:
            tensors_to_save.append(param_tensor)
    torch.save(tensors_to_save, save_path)


def load_model_with_required_grad(model, load_path):
    """Load requires_grad parameters from checkpoint."""
    tensors_to_load = torch.load(load_path, weights_only=False)
    for param_name, param_tensor in model.named_parameters():
        if param_tensor.requires_grad:
            param_tensor.data = tensors_to_load.pop(0).data


# ──────────────────────────────────────────────────────────────────────────────
# Solver
# ──────────────────────────────────────────────────────────────────────────────


class FPMINRSolver:
    """
    Training and inference for FPM-INR.

    Parameters
    ----------
    num_epochs : int
        Number of training epochs.
    lr : float
        Initial learning rate.
    lr_decay_step : int
        StepLR step size.
    lr_decay_gamma : float
        StepLR gamma.
    use_amp : bool
        Use automatic mixed precision (bfloat16).
    use_compile : bool
        Use torch.compile (Linux only).
    """

    def __init__(self, num_epochs=15, lr=1e-3, lr_decay_step=6,
                 lr_decay_gamma=0.1, use_amp=True, use_compile=True):
        self.num_epochs = num_epochs
        self.lr = lr
        self.lr_decay_step = lr_decay_step
        self.lr_decay_gamma = lr_decay_gamma
        self.use_amp = use_amp
        self.use_compile = use_compile

    def train(self, model, forward_model, Isum, z_params, device="cuda:0",
              vis_callback=None):
        """
        Run the full training loop.

        Parameters
        ----------
        model : FullModel
            INR model to optimize.
        forward_model : FPMForwardModel
            Physics forward model.
        Isum : torch.Tensor (M, N, n_leds)
            Normalized measurements on device.
        z_params : dict
            Z-sampling parameters with keys 'z_min', 'z_max', 'num_z'.
        device : str
            Torch device.
        vis_callback : callable, optional
            Called with (epoch, img_ampli, img_phase) for visualization.

        Returns
        -------
        dict with keys:
            'final_loss' : float
            'final_psnr' : float
            'loss_history' : list of float
        """
        z_min = z_params["z_min"]
        z_max = z_params["z_max"]
        num_z = z_params["num_z"]
        ID_len = forward_model.ledpos_true.shape[0]
        led_batch_size = 1

        optimizer = torch.optim.Adam(
            lr=self.lr,
            params=filter(lambda p: p.requires_grad, model.parameters()),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay_gamma
        )

        # Compile model on first epoch
        model_fn = model
        if self.use_compile:
            model_fn = torch.compile(model, backend="inductor")

        loss_history = []
        final_loss = 0.0
        final_psnr = 0.0

        t = tqdm.trange(self.num_epochs)
        for epoch in t:
            led_indices = list(np.arange(ID_len))

            # Z-plane sampling strategy
            dzs = self._sample_z_depths(epoch, z_params, device)

            for dz in dzs:
                dz = dz.unsqueeze(0)

                for it in range(ID_len // led_batch_size):
                    model.zero_grad()

                    led_num = led_indices[
                        it * led_batch_size : (it + 1) * led_batch_size
                    ]
                    spectrum_mask = forward_model.compute_spectrum_mask(dz, led_num)

                    with torch.cuda.amp.autocast(
                        enabled=self.use_amp, dtype=torch.bfloat16
                    ):
                        img_ampli, img_phase = model_fn(dz)
                        img_complex = img_ampli * torch.exp(1j * img_phase)

                        oI_cap = forward_model.get_measured_amplitudes(
                            Isum, led_num, len(dz)
                        )
                        oI_sub = forward_model.get_sub_spectrum(
                            img_complex, led_num, spectrum_mask
                        )

                        l1_loss = F.smooth_l1_loss(oI_cap, oI_sub)
                        loss = l1_loss
                        mse_loss = F.mse_loss(oI_cap, oI_sub)

                    loss.backward()

                    psnr = 10 * -torch.log10(mse_loss).item()
                    t.set_postfix(Loss=f"{loss.item():.4e}", PSNR=f"{psnr:.2f}")
                    optimizer.step()

            scheduler.step()
            final_loss = loss.item()
            final_psnr = psnr
            loss_history.append(final_loss)

            # Visualization callback
            if vis_callback is not None:
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    amplitude = img_ampli[0].float().cpu().detach().numpy()
                    phase = img_phase[0].float().cpu().detach().numpy()
                    vis_callback(epoch, amplitude, phase)

        return {
            "final_loss": final_loss,
            "final_psnr": final_psnr,
            "loss_history": loss_history,
        }

    def _sample_z_depths(self, epoch, z_params, device):
        """
        Z-plane sampling strategy: uniform grid on even epochs,
        random sampling on odd epochs.

        Parameters
        ----------
        epoch : int
        z_params : dict with 'z_min', 'z_max', 'num_z'
        device : str

        Returns
        -------
        dzs : torch.Tensor (num_planes,)
        """
        z_min = z_params["z_min"]
        z_max = z_params["z_max"]
        num_z = z_params["num_z"]

        dzs = (
            (torch.randperm(num_z - 1)[: num_z // 2] + torch.rand(num_z // 2))
            * ((z_max - z_min) // (num_z - 1))
        ).to(device) + z_min

        if epoch % 2 == 0:
            dzs = torch.linspace(z_min, z_max, num_z).to(device)

        return dzs

    def evaluate(self, model, z_positions, device="cuda:0", chunk_size=8):
        """
        Run inference at given z-positions.

        Parameters
        ----------
        model : FullModel
        z_positions : torch.Tensor or ndarray (n_z,)
            Z-positions to evaluate at.
        device : str
        chunk_size : int
            Number of z-planes to process at once.

        Returns
        -------
        amplitude : ndarray (n_z, H, W)
        phase : ndarray (n_z, H, W)
        """
        model.eval()

        if isinstance(z_positions, np.ndarray):
            z_positions = torch.from_numpy(z_positions).float().to(device)
        else:
            z_positions = z_positions.float().to(device)

        with torch.no_grad():
            ampli_list = []
            phase_list = []
            for z_chunk in torch.chunk(
                z_positions, max(1, len(z_positions) // chunk_size)
            ):
                ampli, phase = model(z_chunk)
                ampli_list.append(ampli.float().cpu())
                phase_list.append(phase.float().cpu())

        amplitude = torch.cat(ampli_list, dim=0).numpy()
        phase = torch.cat(phase_list, dim=0).numpy()

        return amplitude, phase
