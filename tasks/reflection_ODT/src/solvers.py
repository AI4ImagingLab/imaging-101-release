"""
Inverse Solvers for Reflection-Mode ODT Reconstruction
========================================================

Reconstruction of the 3D RI distribution from intensity measurements
using FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) with
2D TV regularisation applied per-slice.

The gradient of the data fidelity term is computed automatically via
PyTorch autograd through the entire reflection BPM forward model.

Solver
------
ReflectionBPMReconstructor
    FISTA with amplitude MSE loss, per-slice 2D TV proximal operator,
    and optional non-negativity constraint.

Reference: Zhu et al., "rMS-FPT", arXiv:2503.12246 (2025)
"""

import numpy as np
import torch
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
# FISTA momentum
# ═══════════════════════════════════════════════════════════════════════════

def _fista_q(k: int) -> float:
    """
    Compute the FISTA momentum coefficient q(k).

    q(0) = 1
    q(k) = (1 + sqrt(1 + 4·q(k-1)²)) / 2

    Parameters
    ----------
    k : int
        Iteration index (0-based).

    Returns
    -------
    float
    """
    if k <= 0:
        return 1.0
    q_prev = _fista_q(k - 1)
    return (1.0 + np.sqrt(1.0 + 4.0 * q_prev ** 2)) / 2.0


# ═══════════════════════════════════════════════════════════════════════════
# 2D TV Proximal Operator
# ═══════════════════════════════════════════════════════════════════════════

def _gradient_2d(img: torch.Tensor) -> tuple:
    """Compute forward finite-difference gradient (dx, dy) with zero padding."""
    dx = torch.zeros_like(img)
    dy = torch.zeros_like(img)
    dx[:, :-1] = img[:, 1:] - img[:, :-1]
    dy[:-1, :] = img[1:, :] - img[:-1, :]
    return dx, dy


def _divergence_2d(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """Compute negative divergence (adjoint of gradient)."""
    div = torch.zeros_like(px)
    # x component
    div[:, 0] = px[:, 0]
    div[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    div[:, -1] = -px[:, -2]
    # y component
    div[0, :] += py[0, :]
    div[1:-1, :] += py[1:-1, :] - py[:-2, :]
    div[-1, :] += -py[-2, :]
    return div


def tv_2d_proximal_single(img: torch.Tensor, tau: float,
                          n_iter: int = 20) -> torch.Tensor:
    """
    2D isotropic TV denoising via Chambolle's dual projection algorithm.

    Solves: prox_{τ·TV}(v) = argmin_x { 0.5·‖x - v‖² + τ · TV₂D(x) }

    Parameters
    ----------
    img   : (Ny, Nx) tensor — input image
    tau   : float — TV regularisation strength
    n_iter : int — number of dual iterations

    Returns
    -------
    denoised : (Ny, Nx) tensor
    """
    if tau <= 0:
        return img.clone()

    px = torch.zeros_like(img)
    py = torch.zeros_like(img)

    for _ in range(n_iter):
        # Compute denoised estimate
        div_p = _divergence_2d(px, py)
        x_hat = img - tau * div_p

        # Gradient of denoised estimate
        gx, gy = _gradient_2d(x_hat)

        # Update dual with step size 1/(8*tau)
        px_new = px + gx / (8.0 * tau)
        py_new = py + gy / (8.0 * tau)

        # Project onto unit ball
        norm = torch.sqrt(px_new**2 + py_new**2).clamp(min=1.0)
        px = px_new / norm
        py = py_new / norm

    # Final denoised result
    div_p = _divergence_2d(px, py)
    return img - tau * div_p


def tv_2d_proximal(volume: torch.Tensor, tau: float,
                   n_iter: int = 20) -> torch.Tensor:
    """
    Apply 2D TV proximal operator independently to each z-slice.

    Parameters
    ----------
    volume : (Nz, Ny, Nx) tensor
    tau    : float — TV regularisation strength
    n_iter : int — number of inner dual iterations per slice

    Returns
    -------
    denoised : (Nz, Ny, Nx) tensor
    """
    if tau <= 0:
        return volume.clone()

    result = torch.empty_like(volume)
    for iz in range(volume.shape[0]):
        result[iz] = tv_2d_proximal_single(volume[iz], tau, n_iter)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Reconstructor
# ═══════════════════════════════════════════════════════════════════════════

class ReflectionBPMReconstructor:
    """
    FISTA reconstruction for reflection-mode BPM-ODT.

    Minimises:
        L = Σ_l ‖√I_pred^l − √I_meas^l‖² + τ · Σ_s TV₂D(Δn_s)

    using FISTA acceleration with:
    - Gradient of data fidelity via PyTorch autograd
    - Proximal operator for 2D TV regularisation (per-slice)
    - Optional non-negativity constraint

    Parameters
    ----------
    n_iter : int
        Number of outer iterations.
    lr : float
        Gradient step size (gamma in the paper).
    tv_weight : float
        TV regularisation parameter (tau).
    positivity : bool
        Whether to clamp Δn ≥ 0 after each step.
    device : str
        PyTorch device ('cpu' or 'cuda').
    """

    def __init__(self, n_iter: int = 50, lr: float = 5.0,
                 tv_weight: float = 8e-7, positivity: bool = False,
                 device: str = "cpu"):
        self.n_iter = n_iter
        self.lr = lr
        self.tv_weight = tv_weight
        self.positivity = positivity
        self.device = torch.device(device)

    def reconstruct(self, measurements: torch.Tensor,
                    model) -> tuple:
        """
        Reconstruct 3D RI contrast from amplitude measurements.

        FISTA algorithm:
        1. Initialise dn = 0, x_prev = 0
        2. For each iteration k:
           a. Compute amplitude MSE gradient (summed over all angles)
           b. Gradient step: x_k = dn − lr · grad
           c. TV proximal: x_k = prox_{lr·τ·TV}(x_k)
           d. Optional non-negativity: x_k = max(x_k, 0)
           e. FISTA momentum:
              β = (q(k) − 1) / q(k+1)
              dn = x_k + β · (x_k − x_prev)
           f. Update x_prev = x_k

        Parameters
        ----------
        measurements : (n_angles, Ny, Nx) tensor
            Amplitude measurements (i.e. sqrt(intensity), |field|).
        model : ReflectionBPMForwardModel

        Returns
        -------
        dn_recon : ndarray, shape (Nz, Ny, Nx)
            Reconstructed RI contrast.
        loss_history : list of float
            Loss value at each iteration.
        """
        config = model.config
        nz, ny, nx = config.volume_shape
        n_pixels = ny * nx

        meas = measurements.to(self.device)

        # Initialise RI contrast as zeros
        dn = torch.zeros(nz, ny, nx, dtype=torch.float64,
                         device=self.device, requires_grad=True)
        x_prev = torch.zeros(nz, ny, nx, dtype=torch.float64,
                             device=self.device)

        loss_history = []

        for step in range(self.n_iter):
            total_loss = 0.0

            # Accumulate gradient over all rings and angles
            global_idx = 0
            for ring in config.illumination_rings:
                na = ring["NA"]
                n_ang = ring["n_angles"]
                for m in range(n_ang):
                    pred_intensity = model.forward_single_ring(dn, na, m, n_ang)

                    # Amplitude-domain MSE, normalised per pixel
                    pred_amp = torch.sqrt(pred_intensity + 1e-12)
                    meas_amp = meas[global_idx]
                    loss_m = torch.sum((pred_amp - meas_amp) ** 2) / n_pixels

                    loss_m.backward()
                    total_loss += loss_m.item()
                    global_idx += 1

            loss_history.append(total_loss)
            print(f"  Iteration {step + 1}/{self.n_iter}, loss = {total_loss:.6f}")

            with torch.no_grad():
                # Gradient step
                x_k = dn - self.lr * dn.grad
                dn.grad.zero_()

                # TV proximal operator (per-slice 2D TV)
                if self.tv_weight > 0:
                    x_k = tv_2d_proximal(x_k, self.lr * self.tv_weight)

                # Optional non-negativity constraint
                if self.positivity:
                    x_k.clamp_(min=0.0)

                # FISTA momentum
                q_k = _fista_q(step)
                q_k1 = _fista_q(step + 1)
                beta = (q_k - 1.0) / q_k1

                dn_new = x_k + beta * (x_k - x_prev)
                x_prev.copy_(x_k)
                dn.copy_(dn_new)

        dn_recon = dn.detach().cpu().numpy()
        return dn_recon, loss_history
