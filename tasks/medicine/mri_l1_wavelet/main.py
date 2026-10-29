"""
Multi-Coil MRI Reconstruction with L1-Wavelet Regularization
=============================================================

Pipeline:
    1. Load and preprocess synthetic multi-coil MRI data
    2. Run L1-Wavelet reconstruction via SigPy
    3. Run TV reconstruction for comparison
    4. Evaluate reconstruction quality (NRMSE, NCC, PSNR)
    5. Visualize results and save outputs
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.preprocessing import prepare_data
from src.physics_model import adjoint_operator
from src.solvers import l1_wavelet_reconstruct_batch
from src.visualization import (
    compute_batch_metrics,
    plot_reconstruction_grid,
    plot_error_maps,
    plot_undersampling_mask,
    print_metrics_table,
)

# ── Solver parameters (not in meta_data.json to avoid leaking algorithm info) ──
_L1WAV_LAMBDA = 1e-3
_L1WAV_WAVE_NAME = "db4"


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

    # Ground truth magnitude
    gt_mag = np.abs(ground_truth.squeeze(1))  # (N, H, W)

    # ── Step 2: Zero-filled baseline ──
    print("\nStep 2: Computing zero-filled baseline...")
    zf_images = []
    for i in range(n_samples):
        zf = adjoint_operator(masked_kspace[i], sensitivity_maps[i])
        zf_images.append(np.abs(zf))
    zf_mag = np.stack(zf_images, axis=0)
    zf_metrics = compute_batch_metrics(zf_mag, gt_mag)
    print_metrics_table(zf_metrics, "Zero-Filled")

    # ── Step 3: L1-Wavelet reconstruction ──
    print(f"\nStep 3: Running L1-Wavelet reconstruction (lambda={_L1WAV_LAMBDA}, wavelet={_L1WAV_WAVE_NAME})...")
    wav_recons = l1_wavelet_reconstruct_batch(
        masked_kspace, sensitivity_maps, lamda=_L1WAV_LAMBDA, wave_name=_L1WAV_WAVE_NAME,
    )
    wav_mag = np.abs(wav_recons)
    wav_metrics = compute_batch_metrics(wav_mag, gt_mag)
    print_metrics_table(wav_metrics, "L1-Wavelet")

    # Save L1-wavelet reconstruction
    np.savez_compressed(
        os.path.join(output_dir, "l1_wavelet_reconstruction.npz"),
        reconstruction=wav_recons.astype(np.complex64),
    )

    # TV comparison removed — this task focuses on L1-Wavelet.
    # TV reconstruction is available in the separate mri_tv task.

    # ── Step 5: Save metrics ──
    print("\nStep 5: Saving metrics...")
    metrics_detail = {
        "l1_wavelet": {
            "per_sample": wav_metrics["per_sample"],
            "avg_psnr": wav_metrics["avg_psnr"],
            "avg_ncc": wav_metrics["avg_ncc"],
            "avg_nrmse": wav_metrics["avg_nrmse"],
            "lambda": _L1WAV_LAMBDA,
            "wave_name": _L1WAV_WAVE_NAME,
        },
        "zero_filled": {
            "per_sample": zf_metrics["per_sample"],
            "avg_psnr": zf_metrics["avg_psnr"],
            "avg_ncc": zf_metrics["avg_ncc"],
            "avg_nrmse": zf_metrics["avg_nrmse"],
        },
    }
    with open(os.path.join(output_dir, "metrics_detail.json"), "w") as f:
        json.dump(metrics_detail, f, indent=2)

    # ── Step 6: Visualize ──
    print("\nStep 6: Generating visualizations...")
    plot_reconstruction_grid(
        gt_mag,
        wav_mag,
        zero_filled=zf_mag,
        save_path=os.path.join(output_dir, "reconstruction_grid.png"),
    )
    plot_error_maps(
        gt_mag,
        wav_mag,
        save_path=os.path.join(output_dir, "error_maps.png"),
    )
    plot_undersampling_mask(
        mask,
        save_path=os.path.join(output_dir, "undersampling_mask.png"),
    )
    print("  Saved figures to", output_dir)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Zero-Filled: NCC={zf_metrics['avg_ncc']:.4f}, NRMSE={zf_metrics['avg_nrmse']:.4f}, PSNR={zf_metrics['avg_psnr']:.2f}")
    print(f"  L1-Wavelet:  NCC={wav_metrics['avg_ncc']:.4f}, NRMSE={wav_metrics['avg_nrmse']:.4f}, PSNR={wav_metrics['avg_psnr']:.2f}")
    print("=" * 60)
    print("\nDone!")

    return wav_recons, wav_metrics


if __name__ == "__main__":
    main()
