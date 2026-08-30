"""
Inverse Solvers for SSNP-ODT Reconstruction
=============================================

Reconstruction of the 3D RI distribution from complex field measurements
using gradient-descent optimization with TV regularization.

ODT vs IDT loss
---------------
IDT measures intensity only (|E|²) and minimises amplitude MSE:
    L_IDT = Σ_l ‖|E_pred^l| − √I_meas^l‖²

ODT measures the full complex field E and minimises complex field MSE:
    L_ODT = Σ_l ‖E_pred^l − E_meas^l‖²

The ODT loss is real-valued (squared modulus of a complex residual) and
PyTorch autograd correctly computes gradients of this real scalar with
respect to the real-valued Δn volume through the complex forward model.

Solver
------
SSNPReconstructor
    Gradient descent with complex field MSE loss, 3D TV regularisation,
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
    epsilon : smoothing constant for differentiability

    Returns
    -------
    tv_value : scalar tensor
    """
    dz = volume[1:, :, :] - volume[:-1, :, :]
    dy = volume[:, 1:, :] - volume[:, :-1, :]
    dx = volume[:, :, 1:] - volume[:, :, :-1]

    dz = torch.nn.functional.pad(dz, (0, 0, 0, 0, 0, 1))
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1, 0, 0))
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0, 0, 0))

    return torch.sum(torch.sqrt(dz**2 + dy**2 + dx**2 + epsilon))


# ═══════════════════════════════════════════════════════════════════════════
# Reconstructor
# ═══════════════════════════════════════════════════════════════════════════

class SSNPReconstructor:
    """
    Gradient-descent reconstruction for SSNP-ODT.

    Minimises:
        L = Σ_l ‖E_pred^l − E_meas^l‖² / N_pixels  +  τ · TV(Δn)

    where E_pred and E_meas are complex fields (Ny, Nx). The squared modulus
    ‖·‖² = Σ |·|² is real-valued, giving well-defined real gradients w.r.t.
    the real Δn volume via PyTorch autograd.

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

    def __init__(self, n_iter: int = 100, lr: float = 5.0,
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
        Reconstruct 3D RI contrast from complex field measurements.

        Parameters
        ----------
        measurements : (n_angles, Ny, Nx) complex128 tensor
            Measured complex field images.
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

        meas = measurements.to(self.device)

        dn = torch.zeros(nz, ny, nx, dtype=torch.float64,
                         device=self.device, requires_grad=True)

        loss_history = []

        for step in range(self.n_iter):
            total_loss = 0.0

            for m in range(n_angles):
                E_pred = model.forward_single(dn, m)   # complex (Ny, Nx)
                E_meas = meas[m]                        # complex (Ny, Nx)

                # Complex field MSE, normalised per pixel
                loss_m = torch.sum(torch.abs(E_pred - E_meas) ** 2) / n_pixels
                loss_m.backward()
                total_loss += loss_m.item()

            if self.tv_weight > 0:
                tv_val = self.tv_weight * tv_3d(dn) / (nz * n_pixels)
                tv_val.backward()

            loss_history.append(total_loss)
            print(f"  Iteration {step + 1}/{self.n_iter}, loss = {total_loss:.6f}")

            with torch.no_grad():
                dn -= self.lr * dn.grad
                if self.positivity:
                    dn.clamp_(min=0.0)
                dn.grad.zero_()

        dn_recon = dn.detach().cpu().numpy()
        return dn_recon, loss_history
