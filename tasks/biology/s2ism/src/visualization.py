"""
Visualization utilities for s2ISM results.
"""

import numpy as np


def plot_results(ground_truth, ism_sum, reconstruction, save_path=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(ground_truth, cmap='magma')
    axes[0].set_title('Ground Truth (In-Focus)')
    axes[1].imshow(ism_sum, cmap='magma')
    axes[1].set_title('Raw ISM Sum')
    axes[2].imshow(reconstruction, cmap='magma')
    axes[2].set_title('s2ISM Reconstruction')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    return fig


def compute_metrics(ground_truth, reconstruction, ism_sum):
    """Compute NCC and NRMSE between reconstruction and ground truth."""
    gt_flat = ground_truth.ravel().astype(np.float64)
    recon_flat = reconstruction.ravel().astype(np.float64)
    ism_flat = ism_sum.ravel().astype(np.float64)

    # NCC (cosine similarity, no mean subtraction)
    ncc_recon = float(np.dot(gt_flat, recon_flat) /
                      (np.linalg.norm(gt_flat) * np.linalg.norm(recon_flat)))
    ncc_ism = float(np.dot(gt_flat, ism_flat) /
                    (np.linalg.norm(gt_flat) * np.linalg.norm(ism_flat)))

    # NRMSE (RMS error / dynamic range of reference)
    drange = float(gt_flat.max() - gt_flat.min())
    nrmse_recon = float(np.sqrt(np.mean((recon_flat - gt_flat)**2)) / drange)
    nrmse_ism = float(np.sqrt(np.mean((ism_flat - gt_flat)**2)) / drange)

    return {
        'ncc_recon': ncc_recon,
        'ncc_ism': ncc_ism,
        'nrmse_recon': nrmse_recon,
        'nrmse_ism': nrmse_ism,
    }
