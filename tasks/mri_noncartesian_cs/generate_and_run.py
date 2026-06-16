"""
Generate synthetic data and run reference reconstructions.

This script:
1. Generates synthetic non-Cartesian MRI data (Shepp-Logan + radial sampling)
2. Runs gridding reconstruction (baseline)
3. Runs L1-wavelet CS reconstruction
4. Computes metrics and saves everything
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.generate_data import save_data, generate_synthetic_data
from src.physics_model import gridding_reconstruct, compute_density_compensation
from src.solvers import l1wav_reconstruct_single
from src.visualization import compute_metrics


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(task_dir, "data")
    ref_dir = os.path.join(task_dir, "evaluation", "reference_outputs")
    os.makedirs(ref_dir, exist_ok=True)

    # Step 1: Generate data
    print("Step 1: Generating synthetic data...")
    save_data(data_dir)

    # Step 2: Load back and reconstruct
    print("\nStep 2: Loading data...")
    raw = np.load(os.path.join(data_dir, "raw_data.npz"))
    gt_data = np.load(os.path.join(data_dir, "ground_truth.npz"))

    kdata = raw["kdata"][0].astype(np.complex128)   # (C, M)
    coord = raw["coord"][0].astype(np.float64)       # (M, 2)
    coil_maps = raw["coil_maps"][0].astype(np.complex128)  # (C, H, W)
    phantom = gt_data["phantom"][0].astype(np.complex128)  # (H, W)
    gt_mag = np.abs(phantom)

    image_shape = coil_maps.shape[1:]
    print(f"  kdata: {kdata.shape}, coord: {coord.shape}, coil_maps: {coil_maps.shape}")

    # Step 3: Density compensation
    print("\nStep 3: Computing density compensation...")
    dcf = compute_density_compensation(coord, image_shape, max_iter=30)

    # Step 4: Gridding reconstruction
    print("\nStep 4: Gridding reconstruction...")
    gridding_recon = gridding_reconstruct(kdata, coord, coil_maps, dcf)
    gridding_mag = np.abs(gridding_recon)
    gridding_metrics = compute_metrics(gridding_mag, gt_mag)
    print(f"  Gridding: PSNR={gridding_metrics['psnr']:.2f}dB, "
          f"NCC={gridding_metrics['ncc']:.4f}, NRMSE={gridding_metrics['nrmse']:.4f}")

    # Step 5: L1-Wavelet CS reconstruction
    print("\nStep 5: L1-Wavelet CS reconstruction...")
    l1wav_recon = l1wav_reconstruct_single(
        kdata, coord, coil_maps, lamda=5e-5, max_iter=100
    )
    l1wav_mag = np.abs(l1wav_recon)
    l1wav_metrics = compute_metrics(l1wav_mag, gt_mag)
    print(f"  L1-Wavelet: PSNR={l1wav_metrics['psnr']:.2f}dB, "
          f"NCC={l1wav_metrics['ncc']:.4f}, NRMSE={l1wav_metrics['nrmse']:.4f}")

    # Step 6: Save reference outputs
    print("\nStep 6: Saving reference outputs...")
    np.savez_compressed(
        os.path.join(ref_dir, "gridding_reconstruction.npz"),
        reconstruction=gridding_recon[np.newaxis].astype(np.complex64),
        dcf=dcf[np.newaxis].astype(np.float32),
    )
    np.savez_compressed(
        os.path.join(ref_dir, "l1wav_reconstruction.npz"),
        reconstruction=l1wav_recon[np.newaxis].astype(np.complex64),
    )

    # Step 7: Save metrics
    metrics = {
        "baseline": [
            {
                "method": "L1-Wavelet CS (lambda=5e-5, 100 iter, radial NUFFT)",
                "ncc_vs_ref": round(l1wav_metrics["ncc"], 4),
                "nrmse_vs_ref": round(l1wav_metrics["nrmse"], 4),
            }
        ],
        "ncc_boundary": round(0.9 * l1wav_metrics["ncc"], 4),
        "nrmse_boundary": round(1.1 * l1wav_metrics["nrmse"], 4),
    }

    metrics_path = os.path.join(task_dir, "evaluation", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    # Save detailed metrics for reference
    detail = {
        "gridding": gridding_metrics,
        "l1wav": l1wav_metrics,
        "lambda": 5e-5,
        "max_iter": 100,
    }
    with open(os.path.join(ref_dir, "metrics_detail.json"), "w") as f:
        json.dump(detail, f, indent=2)

    print("\nDone!")
    return gridding_metrics, l1wav_metrics


if __name__ == "__main__":
    main()
