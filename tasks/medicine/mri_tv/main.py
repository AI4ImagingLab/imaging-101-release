"""
Multi-Coil MRI Reconstruction with Total Variation Regularization
=================================================================

Pipeline:
    1. Load and preprocess multi-coil MRI data
    2. Run TV reconstruction via SigPy
    3. Evaluate reconstruction quality (NRMSE, NCC, PSNR)
    4. Visualize results and save outputs
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.preprocessing import prepare_data
from src.physics_model import adjoint_operator
from src.solvers import tv_reconstruct_batch
from src.visualization import (
    compute_batch_metrics,
    plot_reconstruction_grid,
    plot_error_maps,
    plot_undersampling_mask,
    print_metrics_table,
)

# ── Solver parameters (not in meta_data.json to avoid leaking algorithm info) ──
_TV_LAMBDA = 1e-4


def main(data_dir: str = "data", output_dir: str = "evaluation/reference_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load data ──
    print("Step 1: Loading data...")
    obs_data, ground_truth, metadata = prepare_data(data_dir)
    masked_kspace = obs_data["masked_kspace"]
    sensitivity_maps = obs_data["sensitivity_maps"]
    mask = obs_data["undersampling_mask"]
    n_samples = masked_kspace.shape[0]
    print(
        f"  Loaded {n_samples} samples, "
        f"{masked_kspace.shape[1]} coils, "
        f"{masked_kspace.shape[2]}x{masked_kspace.shape[3]} resolution"
    )
    print(
        f"  Undersampling: {int(mask.sum())}/{len(mask)} lines "
        f"({mask.sum() / len(mask) * 100:.1f}%)"
    )

    # ── Step 2: Run TV reconstruction ──
    print(f"\nStep 2: Running TV reconstruction (lambda={_TV_LAMBDA})...")
    recons = tv_reconstruct_batch(masked_kspace, sensitivity_maps, lamda=_TV_LAMBDA)
    print(f"  Reconstructed {recons.shape[0]} images, shape={recons.shape[1:]}")

    # Save reconstructions
    np.savez_compressed(
        os.path.join(output_dir, "tv_reconstruction.npz"),
        reconstruction=recons.astype(np.complex64),
    )

    # ── Step 3: Evaluate ──
    print("\nStep 3: Evaluating reconstruction quality...")
    gt_mag = np.abs(ground_truth.squeeze(1))  # (N, H, W)
    recon_mag = np.abs(recons)  # (N, H, W)
    batch_metrics = compute_batch_metrics(recon_mag, gt_mag)
    print_metrics_table(batch_metrics)

    # Compute zero-filled baseline for comparison
    zf_images = []
    for i in range(n_samples):
        zf = adjoint_operator(masked_kspace[i], sensitivity_maps[i])
        zf_images.append(np.abs(zf))
    zf_mag = np.stack(zf_images, axis=0)
    zf_metrics = compute_batch_metrics(zf_mag, gt_mag)
    print("\nZero-filled baseline:")
    print_metrics_table(zf_metrics)

    # Save metrics
    metrics_out = {
        "tv_reconstruction": {
            "per_sample": batch_metrics["per_sample"],
            "avg_psnr": batch_metrics["avg_psnr"],
            "avg_ncc": batch_metrics["avg_ncc"],
            "avg_nrmse": batch_metrics["avg_nrmse"],
        },
        "zero_filled": {
            "avg_psnr": zf_metrics["avg_psnr"],
            "avg_ncc": zf_metrics["avg_ncc"],
            "avg_nrmse": zf_metrics["avg_nrmse"],
        },
        "lambda": _TV_LAMBDA,
    }
    with open(os.path.join(output_dir, "metrics_detail.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ── Step 4: Visualize ──
    print("\nStep 4: Generating visualizations...")
    plot_reconstruction_grid(
        gt_mag,
        recon_mag,
        zero_filled=zf_mag,
        save_path=os.path.join(output_dir, "reconstruction_grid.png"),
    )
    plot_error_maps(
        gt_mag,
        recon_mag,
        save_path=os.path.join(output_dir, "error_maps.png"),
    )
    plot_undersampling_mask(
        mask,
        save_path=os.path.join(output_dir, "undersampling_mask.png"),
    )
    print("  Saved figures to", output_dir)

    print("\nDone!")
    return recons, batch_metrics


if __name__ == "__main__":
    main()
