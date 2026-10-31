"""
Sparse-View CT Reconstruction
==============================

Reconstructs a 2D phantom from sparse angular projections using:
  1. Filtered Back Projection (FBP) — baseline
  2. TV-regularized Chambolle-Pock (PDHG) — main method

Usage:
    cd tasks/ct_sparse_view
    python main.py
"""

import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")

TASK_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Solver constants (not in meta_data.json per CLAUDE.md) ----
_TV_PARAMS = {
    "lam": 0.01,
    "n_iter": 300,
    "positivity": True,
}


def main():
    from src.preprocessing import load_ground_truth, load_raw_data, load_metadata
    from src.physics_model import filtered_back_projection
    from src.solvers import tv_reconstruction
    from src.visualization import (
        compute_ncc, compute_nrmse, compute_ssim, centre_crop,
        plot_reconstruction_comparison, plot_sinograms, plot_loss_history,
    )

    data_dir = os.path.join(TASK_DIR, "data")
    ref_dir = os.path.join(TASK_DIR, "evaluation", "reference_outputs")
    os.makedirs(ref_dir, exist_ok=True)

    # ---- Step 1: Load data ----
    print("Step 1: Loading data...")
    phantom = load_ground_truth(data_dir)
    raw = load_raw_data(data_dir)
    meta = load_metadata(data_dir)

    sinogram_sparse = raw["sinogram_sparse"]
    sinogram_full = raw["sinogram_full"]
    angles_sparse = raw["angles_sparse"]
    angles_full = raw["angles_full"]
    image_size = meta["image_size"]

    print(f"  Phantom: {phantom.shape}, Sparse sinogram: {sinogram_sparse.shape}")
    print(f"  Full angles: {len(angles_full)}, Sparse angles: {len(angles_sparse)}")

    # ---- Step 2: FBP reconstruction (baseline) ----
    print("Step 2: FBP reconstruction...")
    fbp_full = filtered_back_projection(sinogram_full, angles_full,
                                        output_size=image_size)
    fbp_sparse = filtered_back_projection(sinogram_sparse, angles_sparse,
                                          output_size=image_size)
    print(f"  FBP sparse shape: {fbp_sparse.shape}")

    # ---- Step 3: TV reconstruction ----
    print(f"Step 3: TV reconstruction (lam={_TV_PARAMS['lam']}, "
          f"n_iter={_TV_PARAMS['n_iter']})...")
    tv_recon, loss_history = tv_reconstruction(
        sinogram_sparse, angles_sparse, image_size, **_TV_PARAMS
    )
    print(f"  TV recon shape: {tv_recon.shape}")
    print(f"  Final loss: {loss_history[-1]:.4f}")

    # ---- Step 4: Evaluate metrics ----
    print("Step 4: Computing metrics...")

    # Centre-crop for evaluation
    crop = 0.8
    gt_crop = centre_crop(phantom, crop)
    fbp_crop = centre_crop(fbp_sparse, crop)
    tv_crop = centre_crop(tv_recon, crop)

    # Normalize to [0, 1] range of ground truth
    gt_min, gt_max = gt_crop.min(), gt_crop.max()

    def _normalize(x):
        return (x - gt_min) / (gt_max - gt_min) if gt_max > gt_min else x

    gt_norm = _normalize(gt_crop)
    fbp_norm = _normalize(fbp_crop)
    tv_norm = _normalize(tv_crop)

    fbp_ncc = compute_ncc(fbp_norm, gt_norm)
    fbp_nrmse = compute_nrmse(fbp_norm, gt_norm)
    fbp_ssim = compute_ssim(fbp_crop, gt_crop)

    tv_ncc = compute_ncc(tv_norm, gt_norm)
    tv_nrmse = compute_nrmse(tv_norm, gt_norm)
    tv_ssim = compute_ssim(tv_crop, gt_crop)

    print(f"  FBP  — NCC: {fbp_ncc:.4f}, NRMSE: {fbp_nrmse:.4f}, SSIM: {fbp_ssim:.4f}")
    print(f"  TV   — NCC: {tv_ncc:.4f}, NRMSE: {tv_nrmse:.4f}, SSIM: {tv_ssim:.4f}")

    # ---- Step 5: Save reference outputs ----
    print("Step 5: Saving reference outputs...")

    np.savez(
        os.path.join(ref_dir, "reconstructions.npz"),
        fbp_sparse=fbp_sparse[np.newaxis, ...],
        fbp_full=fbp_full[np.newaxis, ...],
        tv_recon=tv_recon[np.newaxis, ...],
        loss_history=np.array(loss_history),
    )

    # Save metrics.json
    metrics = {
        "baseline": [
            {
                "method": "FBP (30 sparse views)",
                "ncc_vs_ref": round(fbp_ncc, 4),
                "nrmse_vs_ref": round(fbp_nrmse, 4),
                "ssim": round(fbp_ssim, 4),
            },
            {
                "method": "TV-PDHG (30 sparse views, lam=0.01, 300 iter)",
                "ncc_vs_ref": round(tv_ncc, 4),
                "nrmse_vs_ref": round(tv_nrmse, 4),
                "ssim": round(tv_ssim, 4),
            },
        ],
        "ncc_boundary": round(0.9 * tv_ncc, 4),
        "nrmse_boundary": round(1.1 * tv_nrmse, 4),
    }

    metrics_path = os.path.join(TASK_DIR, "evaluation", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Metrics saved to {metrics_path}")

    # ---- Step 6: Visualize ----
    print("Step 6: Generating visualizations...")
    plot_reconstruction_comparison(
        phantom, fbp_sparse, tv_recon,
        save_path=os.path.join(ref_dir, "reconstruction_comparison.png"),
    )
    plot_sinograms(
        sinogram_full, sinogram_sparse, angles_full, angles_sparse,
        save_path=os.path.join(ref_dir, "sinograms.png"),
    )
    plot_loss_history(
        loss_history,
        save_path=os.path.join(ref_dir, "convergence.png"),
    )

    print("Done!")
    return metrics


if __name__ == "__main__":
    main()
