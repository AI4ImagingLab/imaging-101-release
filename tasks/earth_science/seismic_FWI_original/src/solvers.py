"""Full Waveform Inversion optimization loop (no deepwave dependency)."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .physics_model import forward_model


def cosine_taper(x: torch.Tensor, n_taper: int = 5) -> torch.Tensor:
    """Apply a cosine taper to the last n_taper samples of each trace.

    Taper values: (cos(π*i/n_taper) + 1) / 2  for i = 1..n_taper.
    Reduces Gibbs-like edge artifacts at the end of seismic traces.

    Args:
        x: Seismic data tensor, shape (..., nt).
        n_taper: Number of samples to taper at the end. Default 5.

    Returns:
        x_tapered: Tapered tensor, same shape as x.
    """
    if n_taper <= 0:
        return x
    n = min(n_taper, x.shape[-1])
    taper = (
        torch.cos(torch.arange(1, n + 1, device=x.device, dtype=x.dtype) / n * torch.pi)
        + 1.0
    ) / 2.0
    out = x.clone()
    out[..., -n:] = out[..., -n:] * taper
    return out


def smooth_gradient(grad: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Smooth the velocity gradient with a Gaussian filter.

    Gradient smoothing suppresses high-wavenumber artifacts from limited
    illumination and numerical noise.

    Args:
        grad: Gradient array of shape (ny, nx).
        sigma: Gaussian filter sigma in grid points. Default 1.0.

    Returns:
        grad_smoothed: Smoothed gradient, same shape.
    """
    return gaussian_filter(grad, sigma=sigma)


def run_fwi(
    v_init: torch.Tensor,
    spacing: tuple,
    dt: float,
    source_amp: torch.Tensor,
    source_loc: torch.Tensor,
    receiver_loc: torch.Tensor,
    observed_data: torch.Tensor,
    freq: float,
    n_epochs: int = 800,
    lr: float = 1e2,
    milestones: tuple = (75, 300),
    v_min: float = 1480.0,
    v_max: float = 5800.0,
    device: torch.device = torch.device("cpu"),
    print_every: int = 50,
) -> tuple:
    """Run Full Waveform Inversion using Adam with gradient post-processing.

    Minimizes: J(v) = MSE(taper(F(v)), taper(d_obs))

    Each shot is processed sequentially inside forward_model to keep GPU
    memory bounded (one shot's autograd graph at a time).

    Gradient post-processing per iteration:
      1. Gaussian smoothing (sigma=1)
      2. Clip at 98th percentile of |grad|

    Velocity clamped to [v_min, v_max] after each update.

    Args:
        v_init: Initial velocity model, shape (ny, nx), m/s.
        spacing: Grid spacing (dy, dx) in meters.
        dt: User-level time step in seconds.
        source_amp: Source amplitudes, shape (n_shots, 1, nt).
        source_loc: Source positions, shape (n_shots, 1, 2).
        receiver_loc: Receiver positions, shape (n_shots, n_receivers, 2).
        observed_data: Reference seismograms, shape (n_shots, n_receivers, nt).
        freq: Dominant source frequency, Hz.
        n_epochs: Number of gradient iterations. Default 800.
        lr: Adam learning rate. Default 1e2.
        milestones: Epochs to halve the learning rate. Default (75, 300).
        v_min: Minimum velocity bound. Default 1480.0 m/s.
        v_max: Maximum velocity bound. Default 5800.0 m/s.
        device: PyTorch device.
        print_every: Print loss every N epochs. Default 50.

    Returns:
        v_inv: Inverted velocity, shape (ny, nx), m/s (detached, on CPU).
        losses: List of MSE loss values, length n_epochs.
    """
    v1 = v_init.clone().to(device)
    v1.requires_grad_()

    observed_tapered = cosine_taper(observed_data.to(device), n_taper=5)
    loss_fn = torch.nn.MSELoss()

    optimiser = torch.optim.Adam([v1], lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimiser, milestones=list(milestones), gamma=0.5
    )

    losses = []

    for epoch in range(n_epochs):
        optimiser.zero_grad()

        pred_data = forward_model(
            v1, spacing, dt, source_amp, source_loc, receiver_loc, freq
        )
        loss = loss_fn(cosine_taper(pred_data, n_taper=5), observed_tapered)
        loss.backward()

        # Gradient post-processing: smooth then clip
        with torch.no_grad():
            grad_np = v1.grad.detach().cpu().numpy()
            grad_smoothed = smooth_gradient(grad_np, sigma=1.0)
            v1.grad.copy_(torch.tensor(grad_smoothed, dtype=torch.float32).to(device))
            clip_value = torch.quantile(v1.grad.abs(), 0.98)
            torch.nn.utils.clip_grad_value_([v1], clip_value)

        optimiser.step()
        scheduler.step()

        with torch.no_grad():
            v1.clamp_(min=v_min, max=v_max)

        losses.append(loss.item())

        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs}: loss={loss.item():.6e}, "
                f"v=[{v1.min().item():.0f}, {v1.max().item():.0f}] m/s"
            )

    v_inv = v1.detach().cpu()
    return v_inv, losses
