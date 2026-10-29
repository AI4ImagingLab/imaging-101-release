"""Visualization and evaluation metrics for seismic traveltime tomography."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import correlate


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #

def compute_ncc(pred: np.ndarray, ref: np.ndarray) -> float:
    """Cosine-similarity NCC between flattened arrays (no mean subtraction).

    NCC = (pred · ref) / (||pred|| · ||ref||)
    """
    p = pred.ravel().astype(np.float64)
    r = ref.ravel().astype(np.float64)
    denom = np.linalg.norm(p) * np.linalg.norm(r)
    if denom < 1e-30:
        return 0.0
    return float(np.dot(p, r) / denom)


def compute_nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """NRMSE normalised by the dynamic range of the reference.

    NRMSE = RMS(pred − ref) / (max(ref) − min(ref))
    """
    p = pred.ravel().astype(np.float64)
    r = ref.ravel().astype(np.float64)
    rms = np.sqrt(np.mean((p - r) ** 2))
    drange = r.max() - r.min()
    if drange < 1e-30:
        return float('inf')
    return float(rms / drange)


def evaluate_reconstruction(v_inv: np.ndarray,
                             v_true: np.ndarray,
                             v_init: np.ndarray) -> dict:
    """Compute NCC and NRMSE for the recovered velocity perturbation.

    Evaluation is performed on the perturbation field
    (v_inv − v_init) vs (v_true − v_init), which isolates the
    tomographic recovery from the background model.

    Args:
        v_inv:  (Nz, Nx) inverted velocity model.
        v_true: (Nz, Nx) true velocity model.
        v_init: (Nz, Nx) initial (background) velocity model.

    Returns:
        dict with keys 'ncc', 'nrmse', 'ncc_full', 'nrmse_full'.
    """
    dv_inv  = v_inv  - v_init
    dv_true = v_true - v_init
    return {
        'ncc':        compute_ncc(dv_inv, dv_true),
        'nrmse':      compute_nrmse(dv_inv, dv_true),
        'ncc_full':   compute_ncc(v_inv, v_true),
        'nrmse_full': compute_nrmse(v_inv, v_true),
    }


# --------------------------------------------------------------------------- #
#  Plotting helpers                                                            #
# --------------------------------------------------------------------------- #

def _extent(Nx, Nz, dx, dz):
    return [0, (Nx - 1) * dx, (Nz - 1) * dz, 0]


def plot_velocity_model(v: np.ndarray, dx: float, dz: float,
                        title: str = 'Velocity model',
                        vmin: float = None, vmax: float = None,
                        save_path: str = None):
    """Plot a 2D velocity model (x horizontal, z vertical downward)."""
    Nz, Nx = v.shape
    fig, ax = plt.subplots(figsize=(9, 3))
    im = ax.imshow(v, aspect='auto', cmap='seismic_r',
                   extent=_extent(Nx, Nz, dx, dz),
                   vmin=vmin or v.min(), vmax=vmax or v.max())
    plt.colorbar(im, ax=ax, label='Velocity (km/s)')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('z (km)')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_velocity_perturbation(dv: np.ndarray, dx: float, dz: float,
                                title: str = 'Velocity perturbation',
                                vmax: float = None,
                                save_path: str = None):
    """Plot a velocity perturbation (diverging colormap centred at 0)."""
    Nz, Nx = dv.shape
    vmax = vmax or float(np.abs(dv).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(9, 3))
    im = ax.imshow(dv, aspect='auto', cmap='RdBu_r',
                   extent=_extent(Nx, Nz, dx, dz), norm=norm)
    plt.colorbar(im, ax=ax, label='δv (km/s)')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('z (km)')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_checkerboard_recovery(v_true: np.ndarray, v_init: np.ndarray,
                                v_inv: np.ndarray,
                                dx: float, dz: float,
                                sources: np.ndarray = None,
                                receivers: np.ndarray = None,
                                metrics: dict = None,
                                save_path: str = None):
    """Side-by-side comparison: true perturbation vs recovered perturbation."""
    dv_true = v_true - v_init
    dv_inv  = v_inv  - v_init
    vmax = float(np.abs(dv_true).max()) * 1.1

    Nz, Nx = v_true.shape
    ext = _extent(Nx, Nz, dx, dz)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
    for ax, dv, ttl in zip(axes,
                            [dv_true, dv_inv],
                            ['True perturbation', 'Recovered perturbation']):
        im = ax.imshow(dv, aspect='auto', cmap='RdBu_r', extent=ext, norm=norm)
        plt.colorbar(im, ax=ax, label='δv (km/s)')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('z (km)')
        ax.set_title(ttl)
        if sources is not None:
            ax.scatter(sources[:, 0], sources[:, 1],
                       c='k', marker='*', s=20, label='Sources')
        if receivers is not None:
            ax.scatter(receivers[:, 0], receivers[:, 1],
                       c='g', marker='^', s=50, label='Receivers')

    if metrics:
        fig.suptitle(
            f"NCC = {metrics['ncc']:.4f}  |  NRMSE = {metrics['nrmse']:.4f}",
            y=1.01, fontsize=11,
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_convergence(misfit_history: list, save_path: str = None):
    """Plot the misfit convergence curve."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(misfit_history, 'b.-')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Misfit χ (s²)')
    ax.set_title('Traveltime misfit convergence')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_sensitivity_kernel(kernel: np.ndarray, dx: float, dz: float,
                             save_path: str = None):
    """Plot the final normalised sensitivity kernel."""
    Nz, Nx = kernel.shape
    vmax = float(np.abs(kernel).max())
    fig, ax = plt.subplots(figsize=(9, 3))
    im = ax.imshow(kernel, aspect='auto', cmap='RdBu_r',
                   extent=_extent(Nx, Nz, dx, dz),
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Kernel (s/km)')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('z (km)')
    ax.set_title('Normalised sensitivity kernel (final iteration)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
