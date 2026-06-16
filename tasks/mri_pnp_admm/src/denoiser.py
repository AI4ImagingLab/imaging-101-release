"""
RealSN-DnCNN Denoiser for Plug-and-Play MRI
=============================================

Implements a 17-layer DnCNN with Real Spectral Normalization (RealSN),
ensuring the denoiser has Lipschitz constant < 1. This is a residual
denoiser: it estimates the noise component, and the clean image is
obtained by subtraction.

Architecture:
    Layer 1:  Conv(1→64, 3×3) + ReLU
    Layers 2-16: Conv(64→64, 3×3) + BN + ReLU  (15 layers)
    Layer 17: Conv(64→1, 3×3)

Each Conv layer is wrapped with real spectral normalization via
convolution-based power iteration (not matrix-reshape SN).

The per-layer spectral norm is scaled by 0.3^(1/17) ≈ 0.934 to target
a network Lipschitz constant of ~0.3.

Reference
---------
Ryu et al., "Plug-and-Play Methods Provably Converge with Properly
Trained Denoisers," ICML 2019.
"""

import torch
import torch.nn as nn
from torch.nn.functional import conv2d
from torch.nn.parameter import Parameter


def _normalize(tensor, eps=1e-12):
    """L2-normalize a tensor."""
    norm = max(float(torch.sqrt(torch.sum(tensor * tensor))), eps)
    return tensor / norm


class _SpectralNorm:
    """
    Real spectral normalization via convolution-based power iteration.

    Unlike standard SN (which reshapes conv weights to a matrix), this
    computes the spectral norm directly using convolution operations,
    giving a tighter bound for convolutional layers.
    """

    def __init__(self, name="weight", n_power_iterations=1, eps=1e-12):
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")

        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = _normalize(
                    conv2d(u.flip(2, 3), weight.permute(1, 0, 2, 3), padding=2),
                    eps=self.eps,
                ).flip(2, 3)[:, :, 1:-1, 1:-1]
                u = _normalize(conv2d(v, weight, padding=1), eps=self.eps)
                if self.n_power_iterations > 0:
                    u = u.clone()
                    v = v.clone()

        sigma = torch.sum(u * conv2d(v, weight, padding=1))
        weight = weight / sigma
        # Scale for target Lipschitz constant ~0.3 across 17 layers
        weight = weight * pow(0.3, 1.0 / 17.0)
        return weight, u

    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + "_u", u)
        else:
            r_g = getattr(module, self.name + "_orig").requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, eps):
        fn = _SpectralNorm(name, n_power_iterations, eps)
        weight = module._parameters[name]

        C_out = 64 if module.weight.shape[0] != 1 else 1
        u = _normalize(weight.new_empty(1, C_out, 40, 40).normal_(0, 1), eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name="weight", n_power_iterations=1, eps=1e-12):
    """Apply real spectral normalization to a conv layer."""
    _SpectralNorm.apply(module, name, n_power_iterations, eps)
    return module


class RealSN_DnCNN(nn.Module):
    """
    17-layer DnCNN with Real Spectral Normalization.

    Parameters
    ----------
    channels : int
        Number of input/output channels (1 for grayscale).
    num_of_layers : int
        Total number of conv layers (default 17).
    """

    def __init__(self, channels: int = 1, num_of_layers: int = 17):
        super().__init__()
        features = 64
        layers = []

        # First layer: Conv + ReLU
        layers.append(spectral_norm(
            nn.Conv2d(channels, features, 3, padding=1, bias=False)
        ))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: Conv + BN + ReLU
        for _ in range(num_of_layers - 2):
            layers.append(spectral_norm(
                nn.Conv2d(features, features, 3, padding=1, bias=False)
            ))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: Conv only
        layers.append(spectral_norm(
            nn.Conv2d(features, channels, 3, padding=1, bias=False)
        ))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate the noise component of the input.

        Parameters
        ----------
        x : Tensor, (B, 1, H, W)
            Noisy input image.

        Returns
        -------
        noise : Tensor, (B, 1, H, W)
            Estimated noise (residual).
        """
        return self.dncnn(x)


def load_denoiser(weights_path: str, device: str = "cpu") -> nn.Module:
    """
    Load a pretrained RealSN-DnCNN denoiser.

    The pretrained weights were saved with nn.DataParallel wrapping,
    so we load through DataParallel then extract the module.

    Parameters
    ----------
    weights_path : str
        Path to .pth checkpoint file.
    device : str
        Device to load model on.

    Returns
    -------
    model : nn.Module
        Pretrained denoiser in eval mode.
    """
    net = RealSN_DnCNN(channels=1, num_of_layers=17)
    model = nn.DataParallel(net)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
