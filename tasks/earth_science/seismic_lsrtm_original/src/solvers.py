"""Least-Squares Reverse-Time Migration optimization loop (no deepwave)."""

import torch

from .physics_model import forward_model, born_forward_model


def subtract_direct_arrival(
    observed_data: torch.Tensor,
    v_mig: torch.Tensor,
    dx: float,
    dt: float,
    source_amp: torch.Tensor,
    source_loc: torch.Tensor,
    receiver_loc: torch.Tensor,
    freq: float,
    v_true_max: float = None,
) -> torch.Tensor:
    """Estimate and subtract direct arrivals from observed data.

    Args:
        observed_data: Observed seismograms, shape (n_shots, n_rec, nt).
        v_mig: Migration velocity, shape (ny, nx), in m/s.
        dx: Grid spacing in meters.
        dt: Time step in seconds.
        source_amp: Source amplitudes, shape (n_shots, 1, nt).
        source_loc: Source positions, shape (n_shots, 1, 2).
        receiver_loc: Receiver positions, shape (n_shots, n_rec, 2).
        freq: Dominant source frequency, Hz.
        v_true_max: Maximum velocity of true model for CFL matching.

    Returns:
        scattered_data: Observed minus direct arrivals, same shape.
    """
    with torch.no_grad():
        direct = forward_model(
            v_mig, dx, dt,
            source_amp, source_loc, receiver_loc, freq,
            max_vel=v_true_max,
        )
    return observed_data - direct


def run_lsrtm(
    v_mig: torch.Tensor,
    dx: float,
    dt: float,
    source_amp: torch.Tensor,
    source_loc: torch.Tensor,
    receiver_loc: torch.Tensor,
    scattered_data: torch.Tensor,
    freq: float,
    n_epochs: int = 3,
    loss_scale: float = 1e6,
    device: torch.device = torch.device("cpu"),
    print_every: int = 1,
) -> tuple:
    """Run LSRTM inversion using L-BFGS optimizer.

    Args:
        v_mig: Migration velocity, shape (ny, nx), in m/s.
        dx: Grid spacing in meters.
        dt: Time step in seconds.
        source_amp: Source amplitudes, shape (n_shots, 1, nt).
        source_loc: Source positions, shape (n_shots, 1, 2).
        receiver_loc: Receiver positions, shape (n_shots, n_rec, 2).
        scattered_data: Scattered data, shape (n_shots, n_rec, nt).
        freq: Dominant source frequency, Hz.
        n_epochs: Number of L-BFGS iterations. Default 3.
        loss_scale: Scaling factor for loss. Default 1e6.
        device: PyTorch device.
        print_every: Print loss every N epochs. Default 1.

    Returns:
        scatter: Inverted scattering potential, shape (ny, nx) (detached, CPU).
        losses: List of loss values.
    """
    v_mig_dev = v_mig.to(device)
    scattered_data_dev = scattered_data.to(device)

    scatter = torch.zeros_like(v_mig_dev)
    scatter.requires_grad_()

    optimiser = torch.optim.LBFGS([scatter])
    loss_fn = torch.nn.MSELoss()

    losses = []

    for epoch in range(n_epochs):
        closure_loss = [None]

        def closure():
            optimiser.zero_grad()
            pred = born_forward_model(
                v_mig_dev, scatter, dx, dt,
                source_amp, source_loc, receiver_loc, freq,
            )
            loss = loss_scale * loss_fn(pred, scattered_data_dev)
            loss.backward()
            closure_loss[0] = loss.item()
            return loss

        optimiser.step(closure)
        losses.append(closure_loss[0])

        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs}: loss={closure_loss[0]:.6e}, "
                f"scatter range=[{scatter.min().item():.4e}, {scatter.max().item():.4e}]"
            )

    return scatter.detach().cpu(), losses
