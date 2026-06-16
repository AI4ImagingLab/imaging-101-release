"""
Inverse Solvers for SSNP-IDT Reconstruction
=============================================

Reconstruction of the 3D RI distribution from intensity measurements
using gradient-descent optimization with TV regularization.

The gradient is computed automatically via PyTorch autograd through
the entire SSNP forward model (P/Q operators, FFTs, pupil filtering).

Solver
------
SSNPReconstructor
    Gradient descent with amplitude MSE loss, 3D TV regularisation,
    and non-negativity constraint (projection after each step).
"""

import numpy as np
import torch
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
# TV Regularisation
# ═══════════════════════════════════════════════════════════════════════════

def tv_3d(volume: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    3D isotropic total variation (Huber-smoothed).

    TV(x) = Σ_i sqrt(|∂x/∂z|² + |∂x/∂y|² + |∂x/∂x|² + ε)

    Parameters
    ----------
    volume  : (Nz, Ny, Nx) real tensor
    epsilon : float — smoothing constant for differentiability

    Returns
    -------
    tv_value : scalar tensor
    """
    dz = volume[1:, :, :] - volume[:-1, :, :]
    dy = volume[:, 1:, :] - volume[:, :-1, :]
    dx = volume[:, :, 1:] - volume[:, :, :-1]

    # Pad to uniform size (replicate last slice along each axis)
    dz = torch.nn.functional.pad(dz, (0, 0, 0, 0, 0, 1))
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1, 0, 0))
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0, 0, 0))

    return torch.sum(torch.sqrt(dz**2 + dy**2 + dx**2 + epsilon))


# ═══════════════════════════════════════════════════════════════════════════
# Reconstructor
# ═══════════════════════════════════════════════════════════════════════════

class SSNPReconstructor:
    """
    Gradient-descent reconstruction for SSNP-IDT.

    Minimises:
        L = Σ_l ‖√I_pred^l − √I_meas^l‖² + τ · TV(Δn)

    using PyTorch autograd for gradient computation and a fixed learning
    rate with non-negativity projection.

    Parameters
    ----------
    n_iter : int
        Number of gradient-descent iterations.
    lr : float
        Learning rate (gradient step size).
    tv_weight : float
        Weight τ for 3D total variation regularisation.
    positivity : bool
        Whether to project Δn ≥ 0 after each step.
    device : str
        PyTorch device ('cpu' or 'cuda').
    """

    def __init__(self, n_iter: int = 5, lr: float = 350.0,
                 tv_weight: float = 0.0, positivity: bool = True,
                 device: str = "cpu"):
        self.n_iter = n_iter
        self.lr = lr
        self.tv_weight = tv_weight
        self.positivity = positivity
        self.device = torch.device(device)

    def reconstruct(self, measurements: torch.Tensor,
                    model) -> tuple:
        """
        Reconstruct 3D RI contrast from intensity measurements.

        Follows the same gradient-descent scheme as the original SSNP-IDT
        code: for each iteration, accumulate per-angle gradients (each
        normalised by N_pixels), then apply a single step.

        Parameters
        ----------
        measurements : (n_angles, Ny, Nx) tensor
            Measured intensity images (amplitudes, i.e. |field|).
        model : SSNPForwardModel
            Forward model instance.

        Returns
        -------
        dn_recon : ndarray, shape (Nz, Ny, Nx)
            Reconstructed RI contrast.
        loss_history : list of float
            Loss value at each iteration.
        """
        config = model.config
        nz, ny, nx = config.volume_shape
        n_angles = config.n_angles
        n_pixels = ny * nx

        # measurements should be amplitudes (|field|) to match original code
        meas = measurements.to(self.device)

        # Initialise RI contrast as zeros
        dn = torch.zeros(nz, ny, nx, dtype=torch.float64,
                         device=self.device, requires_grad=True)

        loss_history = []

        for step in range(self.n_iter):
            total_loss = 0.0

            # Accumulate gradient over all angles (matching original code)
            for m in range(n_angles):
                # Forward model for single angle
                pred_intensity = model.forward_single(dn, m)

                # Amplitude-domain MSE, normalised per pixel
                # loss_m = ||abs(field) - meas_m||² / N_pixels
                pred_amp = torch.sqrt(pred_intensity + 1e-12)
                meas_amp = meas[m]
                loss_m = torch.sum((pred_amp - meas_amp) ** 2) / n_pixels

                loss_m.backward()
                total_loss += loss_m.item()

            # Add TV regularisation gradient
            if self.tv_weight > 0:
                tv_val = self.tv_weight * tv_3d(dn) / (nz * n_pixels)
                tv_val.backward()

            loss_history.append(total_loss)
            print(f"  Iteration {step + 1}/{self.n_iter}, loss = {total_loss:.6f}")

            # Gradient descent step
            with torch.no_grad():
                dn -= self.lr * dn.grad
                if self.positivity:
                    dn.clamp_(min=0.0)
                dn.grad.zero_()

        dn_recon = dn.detach().cpu().numpy()
        return dn_recon, loss_history
