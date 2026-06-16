"""Visualisation utilities for the Shack-Hartmann wavefront sensing task."""

import numpy as np
import matplotlib.pyplot as plt


def plot_wfs_image(
    image: np.ndarray,
    det_shape: tuple,
    title: str = 'SH-WFS detector image',
    output_path: str = None,
):
    """Display the raw SH-WFS detector image as a 2D spot array.

    Parameters
    ----------
    image      : (N_det,) or (H, W)  detector image [photons]
    det_shape  : (H, W)    reshape target
    title      : str
    """
    H, W  = det_shape
    img2d = image.reshape(H, W)
    # log scale to reveal faint spots
    img_log = np.log1p(np.maximum(img2d, 0))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_log, origin='lower', cmap='inferno', aspect='equal')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x pixel', fontsize=9)
    ax.set_ylabel('y pixel', fontsize=9)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_wavefront_comparison(
    gt_phases: np.ndarray,
    recon_phases: np.ndarray,
    aperture: np.ndarray,
    pupil_shape: tuple,
    wfe_levels_nm: list,
    ncc_arr: np.ndarray = None,
    nrmse_arr: np.ndarray = None,
    output_path: str = None,
):
    """Grid of ground-truth / reconstructed / error wavefront maps.

    Parameters
    ----------
    gt_phases    : (N_levels, N_pupil_px)
    recon_phases : (N_levels, N_pupil_px)
    aperture     : (N_pupil_px,)
    pupil_shape  : (H, W)
    wfe_levels_nm: list of N_levels ints [nm]
    ncc_arr      : (N_levels,) optional
    nrmse_arr    : (N_levels,) optional
    """
    H, W     = pupil_shape
    N_levels = len(wfe_levels_nm)
    mask_2d  = aperture.reshape(H, W) > 0.5

    fig, axes = plt.subplots(N_levels, 3, figsize=(11, 3 * N_levels))
    if N_levels == 1:
        axes = axes[np.newaxis]

    for j, ct in enumerate(['Ground truth φ [rad]',
                             'Reconstructed φ̂ [rad]',
                             'Error φ̂ − φ [rad]']):
        axes[0, j].set_title(ct, fontsize=10)

    for i, wfe in enumerate(wfe_levels_nm):
        gt  = np.where(mask_2d, gt_phases[i].reshape(H, W),    np.nan)
        rec = np.where(mask_2d, recon_phases[i].reshape(H, W), np.nan)
        err = rec - gt

        lim  = np.nanpercentile(np.abs(gt),  99)
        elim = np.nanpercentile(np.abs(err), 99)

        for j, (img, vlim) in enumerate([(gt, lim), (rec, lim), (err, elim)]):
            im = axes[i, j].imshow(img, origin='lower',
                                   cmap='RdBu_r', vmin=-vlim, vmax=vlim)
            axes[i, j].axis('off')
            plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)

        label = f'WFE = {wfe} nm'
        if ncc_arr is not None:
            label += f'\nNCC={ncc_arr[i]:.4f}  NRMSE={nrmse_arr[i]:.4f}'
        axes[i, 0].set_ylabel(label, fontsize=9, rotation=0,
                               labelpad=110, va='center')

    plt.suptitle('Shack-Hartmann Wavefront Reconstruction', fontsize=12, y=1.01)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_metrics_vs_wfe(
    wfe_levels_nm: list,
    ncc_arr: np.ndarray,
    nrmse_arr: np.ndarray,
    ncc_boundaries: list = None,
    nrmse_boundaries: list = None,
    output_path: str = None,
):
    """NCC and NRMSE vs WFE level (semi-log for NRMSE)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(wfe_levels_nm, ncc_arr, 'o-', lw=2, ms=8, color='steelblue')
    if ncc_boundaries is not None:
        for wfe, bd in zip(wfe_levels_nm, ncc_boundaries):
            ax1.axhline(bd, color='r', ls='--', lw=0.9, alpha=0.6)
    ax1.set_xlabel('WFE level [nm]')
    ax1.set_ylabel('NCC')
    ax1.set_title('NCC vs WFE level')
    ax1.set_ylim(0.5, 1.02)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(wfe_levels_nm, nrmse_arr, 'o-', lw=2, ms=8, color='tomato')
    if nrmse_boundaries is not None:
        for wfe, bd in zip(wfe_levels_nm, nrmse_boundaries):
            ax2.axhline(bd, color='r', ls='--', lw=0.9, alpha=0.6)
    ax2.set_xlabel('WFE level [nm]')
    ax2.set_ylabel('NRMSE')
    ax2.set_title('NRMSE vs WFE level')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_dm_modes(
    dm_modes: np.ndarray,
    aperture: np.ndarray,
    pupil_shape: tuple,
    n_show: int = 6,
    output_path: str = None,
):
    """Plot the first n_show DM mode shapes."""
    H, W = pupil_shape
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    for i, ax in enumerate(axes):
        mode = dm_modes[i].reshape(H, W)
        aper = aperture.reshape(H, W)
        img  = np.where(aper > 0, mode, np.nan)
        lim  = np.nanpercentile(np.abs(img), 99)
        ax.imshow(img, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_title(f'Mode {i + 1}', fontsize=9)
        ax.axis('off')
    plt.suptitle('DM Mode Shapes (Disk Harmonics)', fontsize=11)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_response_singular_values(
    response_matrix: np.ndarray,
    rcond: float = 1e-3,
    output_path: str = None,
):
    """Plot singular values of the response matrix."""
    _, s, _ = np.linalg.svd(response_matrix, full_matrices=False)
    n_kept  = int((s / s[0] > rcond).sum())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(s / s[0], lw=1.5, color='steelblue', label='Singular values')
    ax.axhline(rcond, color='r', ls='--', lw=1.2,
               label=f'rcond={rcond:.0e}  ({n_kept}/{len(s)} modes kept)')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Normalised singular value')
    ax.set_title('Response matrix singular values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig
