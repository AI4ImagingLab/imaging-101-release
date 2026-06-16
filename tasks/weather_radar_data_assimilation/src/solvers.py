"""Flow-based data assimilation solver with guided Euler-Maruyama sampling.

Adapted from FlowDAS (Chen et al., 2025). The solver uses a pretrained UNet
drift model within a stochastic interpolant framework, guided by sparse
observations through gradient-based likelihood correction.
"""

import math
from functools import partial
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─── UNet architecture (from lucidrains) ────────────────────────────────────

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, classes_emb_dim=None, groups=8):
        super().__init__()
        int_time_emb_dim = int(time_emb_dim)
        int_classes_emb_dim = int(classes_emb_dim) if classes_emb_dim is not None else 0
        int_both = int_time_emb_dim + int_classes_emb_dim
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(int_both, dim_out * 2))
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, "b c -> b c 1 1")
            scale_shift = cond_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        out_channels,
        dim=128,
        dim_mults=(1, 2, 2, 2),
        resnet_block_groups=8,
        learned_sinusoidal_cond=True,
        random_fourier_features=False,
        learned_sinusoidal_dim=32,
        attn_dim_head=64,
        attn_heads=4,
        use_classes=True,
    ):
        super().__init__()
        self.use_classes = use_classes
        self.in_channels = in_channels
        self.init_conv = nn.Conv2d(in_channels, dim, 7, padding=3)
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        if self.use_classes:
            self.classes_emb = nn.Embedding(num_classes, dim)
            classes_dim = dim * 4
            self.classes_mlp = nn.Sequential(
                nn.Linear(dim, classes_dim), nn.GELU(), nn.Linear(classes_dim, classes_dim)
            )
        else:
            classes_dim = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_channels = out_channels
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_channels, 1)

    def forward(self, x, time, classes=None):
        if classes is not None:
            c = self.classes_mlp(self.classes_emb(classes))
        else:
            c = None
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)
            x = block2(x, t, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t, c)
        return self.final_conv(x)


# ─── Drift model wrapper ────────────────────────────────────────────────────


class DriftModel(nn.Module):
    """UNet-based drift model for the stochastic interpolant."""

    def __init__(self, in_channels=4, out_channels=1, unet_channels=128,
                 dim_mults=(1, 2, 2, 2), use_classes=False, num_classes=1):
        super().__init__()
        self.use_classes = use_classes
        self._arch = Unet(
            num_classes=num_classes,
            in_channels=in_channels,
            out_channels=out_channels,
            dim=unet_channels,
            dim_mults=dim_mults,
            resnet_block_groups=8,
            learned_sinusoidal_cond=True,
            random_fourier_features=False,
            learned_sinusoidal_dim=32,
            attn_dim_head=64,
            attn_heads=4,
            use_classes=use_classes,
        )

    def forward(self, zt, t, y, cond=None):
        if not self.use_classes:
            y = None
        if cond is not None:
            zt = torch.cat([zt, cond], dim=1)
        return self._arch(zt, t, y)


# ─── Stochastic interpolant ─────────────────────────────────────────────────


class StochasticInterpolant:
    """Defines interpolant coefficients for flow matching with beta(t) = t^2."""

    def __init__(self, beta_fn="t^2", sigma_coef=1.0):
        self.beta_fn = beta_fn
        self.sigma_coef = sigma_coef

    def _wide(self, t):
        return t[:, None, None, None]

    def alpha(self, t):
        return self._wide(1 - t)

    def beta(self, t):
        return self._wide(t.pow(2) if self.beta_fn == "t^2" else t)

    def sigma(self, t):
        return self.sigma_coef * self._wide(1 - t)


# ─── Guided EM sampling ─────────────────────────────────────────────────────


def _taylor_est_2nd_order(model, xt, t, bF, label, cond, mc_times):
    """Second-order stochastic Runge-Kutta estimate of x_1."""
    variance_term = 2.0 / 3.0 - t.sqrt() + (1.0 / 3.0) * (t.sqrt()) ** 3
    hat_x1 = xt + bF * (1 - t) + torch.randn_like(xt) * variance_term
    t1 = torch.FloatTensor([1]).to(xt.device)
    bF2 = model(hat_x1, t1, label, cond=cond).requires_grad_(True)
    if mc_times == 1:
        hat_x1 = xt + (bF + bF2) / 2 * (1 - t) + torch.randn_like(xt) * variance_term
        return hat_x1.requires_grad_(True)
    else:
        hat_x1_list = []
        for _ in range(mc_times):
            hat_x1 = xt + (bF + bF2) / 2 * (1 - t) + torch.randn_like(xt) * variance_term
            hat_x1_list.append(hat_x1.requires_grad_(True))
        return hat_x1_list


def _grad_and_value(x_prev, x_0_hat, measurement, operator, noiser):
    """Compute gradient of observation likelihood for guidance."""
    if isinstance(x_0_hat, torch.Tensor):
        difference = (measurement - noiser(operator(x_0_hat))).requires_grad_(True)
        norm = torch.linalg.norm(difference).requires_grad_(True)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, allow_unused=True)[0]
    else:
        difference = sum(
            (measurement - operator(x_0_hat[i])).requires_grad_(True)
            for i in range(len(x_0_hat))
        ) / len(x_0_hat)
        norm = torch.linalg.norm(difference).requires_grad_(True)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, allow_unused=True)[0]
    return norm_grad, norm


def guided_em_sample(
    model,
    interpolant,
    base,
    cond,
    observation,
    operator,
    noiser,
    n_steps=500,
    mc_times=25,
    guidance_scale=0.1,
    device="cuda",
):
    """Run guided Euler-Maruyama sampling to reconstruct a single frame.

    Parameters
    ----------
    model : DriftModel
        Pretrained drift model.
    interpolant : StochasticInterpolant
        Interpolant defining the flow coefficients.
    base : torch.Tensor
        Base sample (last conditioning frame), shape (B, 1, H, W).
    cond : torch.Tensor
        Full conditioning frames, shape (B, 6, H, W).
    observation : torch.Tensor
        Sparse observation for this frame, shape (B, 1, H, W).
    operator : callable
        Forward measurement operator (masking).
    noiser : callable
        Noise model.
    n_steps : int
        Number of EM time steps.
    mc_times : int
        Monte Carlo samples for variance reduction.
    guidance_scale : float
        Step size for observation guidance gradient.
    device : str
        Torch device.

    Returns
    -------
    torch.Tensor
        Reconstructed frame, shape (B, 1, H, W).
    """
    tmin, tmax = 0.0, 0.999
    ts = torch.linspace(tmin, tmax, n_steps)
    dt = ts[1] - ts[0]
    ones = torch.ones(base.shape[0])

    xt = base.requires_grad_(True)
    label = None  # no class conditioning for weather

    for i, tscalar in enumerate(ts):
        t = tscalar * ones
        t_tensor = t.clone().detach().to(device)

        bF = model(xt, t_tensor, label, cond=cond).requires_grad_(True)
        sigma = interpolant.sigma(t_tensor)

        f = bF
        g = sigma

        es_x1 = _taylor_est_2nd_order(model, xt, interpolant._wide(t_tensor), bF, label, cond, mc_times)
        norm_grad, norm = _grad_and_value(xt, es_x1, observation, operator, noiser)

        mu = xt + f * dt
        if norm_grad is None:
            norm_grad = 0
        xt = mu + g * torch.randn_like(mu) * dt.sqrt() - guidance_scale * norm_grad

        if (i + 1) % 100 == 0:
            print(f"  EM step {i+1}/{n_steps}, guidance norm: {norm.item():.4f}")

    return mu  # return mean (denoised)


def autoregressive_reconstruct(
    model,
    interpolant,
    condition_frames,
    observations,
    operator,
    noiser,
    n_steps=500,
    mc_times=25,
    auto_steps=3,
    device="cuda",
):
    """Autoregressively reconstruct multiple future frames.

    Parameters
    ----------
    model : DriftModel
        Pretrained drift model.
    interpolant : StochasticInterpolant
        Interpolant.
    condition_frames : torch.Tensor
        Past frames, shape (B, 6, H, W) in model scale.
    observations : torch.Tensor
        Sparse observations of future frames, shape (B, auto_steps, H, W) in model scale.
    operator : callable
        Forward measurement operator.
    noiser : callable
        Noise model.
    n_steps : int
        EM steps per frame.
    mc_times : int
        MC samples for guidance.
    auto_steps : int
        Number of frames to predict.
    device : str
        Torch device.

    Returns
    -------
    torch.Tensor
        Reconstructed frames, shape (B, auto_steps, H, W).
    """
    cond = condition_frames.clone()
    results = []

    for step in range(auto_steps):
        print(f"Autoregressive step {step+1}/{auto_steps}")
        base = cond[:, -1:, :, :]  # last frame as base
        obs_step = observations[:, step:step+1, :, :]

        sample = guided_em_sample(
            model, interpolant, base, cond, obs_step,
            operator, noiser, n_steps, mc_times,
            device=device,
        )
        results.append(sample)

        if step < auto_steps - 1:
            cond = torch.cat([cond[:, 1:], sample], dim=1)

    return torch.cat(results, dim=1)


def load_drift_model(checkpoint_path, device="cuda"):
    """Load a pretrained drift model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt checkpoint file.
    device : str
        Device to load to.

    Returns
    -------
    DriftModel
        Loaded model in eval mode.
    """
    # The UNet receives [zt (1ch), cond (6ch)] = 7 input channels.
    # The DriftModel.forward concatenates zt and cond before passing to the UNet.
    model = DriftModel(
        in_channels=7,  # 1 (current state) + 6 (conditioning frames)
        out_channels=1,
        unet_channels=128,
        dim_mults=(1, 2, 2, 2),
        use_classes=False,
        num_classes=1,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}, step {checkpoint['step']}")
    return model
