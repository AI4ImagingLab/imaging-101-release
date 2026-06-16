"""
Visualization utilities for differentiable refractive deflectometry.

Extracted from: diffmetrology/utils.py (spot_diagram, imshow, ploterror)
and demo_experiments.py (visualize, loss curve plotting).
"""
import json

import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_spot_diagram(ps_measured, ps_modeled, valid, camera_count, filename=None):
    """Plot measurement vs modeled spot diagrams.

    Args:
        ps_measured: measured intersection points (Tensor) [N_cam, H, W, 3]
        ps_modeled: modeled intersection points (Tensor) [N_cam, H, W, 3]
        valid: validity mask (Tensor) [N_cam, H, W]
        camera_count: number of cameras
        filename: optional save path
    """
    figure, ax = plt.subplots(1, camera_count, figsize=(6 * camera_count, 5))
    if camera_count == 1:
        ax = [ax]

    def sub_sampling(x):
        Ns = [8, 8]
        return x[::Ns[0], ::Ns[1], ...]

    for i in range(camera_count):
        mask = sub_sampling(valid[i])
        ref = sub_sampling(ps_measured[i])[mask].cpu().detach().numpy()
        cap = sub_sampling(ps_modeled[i])[mask].cpu().detach().numpy()
        ax[i].plot(ref[..., 0], ref[..., 1], 'b.', label='Measurement')
        ax[i].plot(cap[..., 0], cap[..., 1], 'r.', label='Modeled (reprojection)')
        ax[i].legend()
        ax[i].set_xlabel('[mm]')
        ax[i].set_ylabel('[mm]')
        ax[i].set_aspect(1)
        ax[i].set_title('camera ' + str(i + 1))

    figure.suptitle('Spot Diagram')
    if filename:
        figure.savefig(filename, bbox_inches='tight')
    return figure


def plot_image_comparison(I_measured, I_modeled, valid, filename_prefix=None):
    """Plot measurement/modeled/error images side by side.

    Args:
        I_measured: measured images (Tensor) [N_cam, H, W]
        I_modeled: modeled images (Tensor) [N_cam, H, W]
        valid: validity mask (used for display)
        filename_prefix: prefix for saved files (e.g. 'initial' -> 'initial0.jpg')
    """
    n_cams = I_measured.shape[0]
    fig, axes = plt.subplots(n_cams, 3, figsize=(12, 4 * n_cams))
    if n_cams == 1:
        axes = axes[None, :]

    for i in range(n_cams):
        im = axes[i, 0].imshow(I_measured[i].cpu(), vmin=0, vmax=1, cmap='gray')
        axes[i, 0].set_title(f"Camera {i + 1}\nMeasurement")
        axes[i, 0].set_xlabel('[pixel]')
        axes[i, 0].set_ylabel('[pixel]')
        plt.colorbar(im, ax=axes[i, 0])

        im = axes[i, 1].imshow(I_modeled[i].cpu().detach(), vmin=0, vmax=1, cmap='gray')
        plt.colorbar(im, ax=axes[i, 1])
        axes[i, 1].set_title(f"Camera {i + 1}\nModeled")
        axes[i, 1].set_xlabel('[pixel]')
        axes[i, 1].set_ylabel('[pixel]')

        im = axes[i, 2].imshow(I_measured[i].cpu() - I_modeled[i].cpu().detach(), vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar(im, ax=axes[i, 2])
        axes[i, 2].set_title(f"Camera {i + 1}\nError")
        axes[i, 2].set_xlabel('[pixel]')
        axes[i, 2].set_ylabel('[pixel]')

    if filename_prefix:
        for i in range(n_cams):
            fig.savefig(filename_prefix + str(i) + ".jpg", bbox_inches='tight')
    return fig


def plot_loss_curve(loss_history, filename=None):
    """Plot optimization loss on semilogy scale.

    Args:
        loss_history: list or array of loss values
        filename: optional save path
    """
    fig = plt.figure()
    plt.semilogy(loss_history, '-o', color='k')
    plt.xlabel('LM iteration')
    plt.ylabel('Loss')
    plt.title("Optimization Loss")
    if filename:
        fig.savefig(filename, bbox_inches='tight')
    return fig


def compute_metrics(ps_modeled, ps_measured, valid, loss_history, recovered_params, gt_params):
    """Compute all evaluation metrics.

    Args:
        ps_modeled: modeled intersection points (Tensor)
        ps_measured: measured intersection points (Tensor)
        valid: validity mask (Tensor)
        loss_history: list of loss values
        recovered_params: dict of recovered parameter values
        gt_params: dict of ground truth parameter values

    Returns:
        metrics: dict with all evaluation metrics
    """
    T = ps_modeled - ps_measured
    disp_error = torch.sqrt(torch.sum(T[valid, ...]**2, axis=-1)).mean()

    metrics = {
        "final_loss": float(loss_history[-1]),
        "n_iterations": len(loss_history),
        "mean_displacement_error_um": float(disp_error * 1e3),
    }

    # Add recovered parameters
    for key, val in recovered_params.items():
        metrics[f"recovered_{key}"] = float(val)

    # Add relative errors vs ground truth
    for key in gt_params:
        if key in recovered_params:
            gt = gt_params[key]
            rec = recovered_params[key]
            if gt != 0:
                metrics[f"relative_error_{key}"] = abs(rec - gt) / abs(gt)

    return metrics


def save_metrics(metrics, filepath):
    """Save metrics dict to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
