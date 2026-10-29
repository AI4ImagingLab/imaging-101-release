"""
Low-Dose CT Reconstruction with Poisson Noise
==============================================

Reconstructs a 2D Shepp-Logan phantom from Poisson-noise sinograms using:
  1. Unweighted TV (baseline -- ignores noise statistics)
  2. SVMBIR PWLS with Poisson-derived weights (main method)

Usage:
    cd tasks/ct_poisson_lowdose
    python main.py
"""

import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")

TASK_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    from src.preprocessing import load_ground_truth, load_raw_data, load_metadata
    from src.solvers import unweighted_tv_reconstruction, pwls_tv_reconstruction
    from src.visualization import (
        compute_ncc, compute_nrmse, centre_crop,
        plot_reconstruction_comparison, plot_sinogram_comparison,
        plot_dose_comparison, plot_error_maps,
    )

    data_dir = os.path.join(TASK_DIR, "data")
    ref_dir = os.path.join(TASK_DIR, "evaluation", "reference_outputs")
    os.makedirs(ref_dir, exist_ok=True)

    # ---- Step 1: Load data ----
    print("Step 1: Loading data...")
    phantom = load_ground_truth(data_dir)
    raw = load_raw_data(data_dir)
    meta = load_metadata(data_dir)

    sino_clean = raw["sinogram_clean"]
    sino_low = raw["sinogram_low_dose"]
    sino_high = raw["sinogram_high_dose"]
    weights_low = raw["weights_low_dose"]
    weights_high = raw["weights_high_dose"]
    angles = raw["angles"]
    image_size = meta["image_size"]

    print(f"  Phantom: {phantom.shape}")
    print(f"  Low-dose sinogram: {sino_low.shape}")
    print(f"  High-dose sinogram: {sino_high.shape}")

    # ---- Step 2: Unweighted TV reconstruction (baseline) ----
    print("Step 2: Unweighted TV reconstruction (low-dose)...")
    recon_unweighted = unweighted_tv_reconstruction(
        sino_low, angles, image_size, image_size)
    print(f"  Unweighted recon shape: {recon_unweighted.shape}")

    # ---- Step 3: PWLS-TV reconstruction ----
    print("Step 3: PWLS-TV reconstruction (low-dose)...")
    recon_pwls_low = pwls_tv_reconstruction(
        sino_low, angles, weights_low, image_size, image_size)
    print(f"  PWLS low-dose recon shape: {recon_pwls_low.shape}")

    print("  PWLS-TV reconstruction (high-dose)...")
    recon_pwls_high = pwls_tv_reconstruction(
        sino_high, angles, weights_high, image_size, image_size)
    print(f"  PWLS high-dose recon shape: {recon_pwls_high.shape}")

    # ---- Step 4: Evaluate metrics ----
    print("Step 4: Computing metrics...")

    crop_frac = 0.8
    gt_crop = centre_crop(phantom, crop_frac)
    uw_crop = centre_crop(recon_unweighted, crop_frac)
    pwls_low_crop = centre_crop(recon_pwls_low, crop_frac)
    pwls_high_crop = centre_crop(recon_pwls_high, crop_frac)

    # Metrics on centre crop
    uw_ncc = compute_ncc(uw_crop, gt_crop)
    uw_nrmse = compute_nrmse(uw_crop, gt_crop)

    pwls_low_ncc = compute_ncc(pwls_low_crop, gt_crop)
    pwls_low_nrmse = compute_nrmse(pwls_low_crop, gt_crop)

    pwls_high_ncc = compute_ncc(pwls_high_crop, gt_crop)
    pwls_high_nrmse = compute_nrmse(pwls_high_crop, gt_crop)

    print(f"  Unweighted (low-dose)  -- NCC: {uw_ncc:.4f}, NRMSE: {uw_nrmse:.4f}")
    print(f"  PWLS (low-dose)        -- NCC: {pwls_low_ncc:.4f}, NRMSE: {pwls_low_nrmse:.4f}")
    print(f"  PWLS (high-dose)       -- NCC: {pwls_high_ncc:.4f}, NRMSE: {pwls_high_nrmse:.4f}")

    # ---- Step 5: Save reference outputs ----
    print("Step 5: Saving reference outputs...")

    np.savez(
        os.path.join(ref_dir, "reconstructions.npz"),
        recon_unweighted=recon_unweighted[np.newaxis, ...],   # (1, H, W)
        recon_pwls_low=recon_pwls_low[np.newaxis, ...],       # (1, H, W)
        recon_pwls_high=recon_pwls_high[np.newaxis, ...],     # (1, H, W)
    )

    # Save metrics.json
    metrics = {
        "baseline": [
            {
                "method": "Unweighted TV (low-dose, I0=1000)",
                "ncc_vs_ref": round(uw_ncc, 4),
                "nrmse_vs_ref": round(uw_nrmse, 4),
            },
            {
                "method": "PWLS-TV (low-dose, I0=1000)",
                "ncc_vs_ref": round(pwls_low_ncc, 4),
                "nrmse_vs_ref": round(pwls_low_nrmse, 4),
            },
            {
                "method": "PWLS-TV (high-dose, I0=50000)",
                "ncc_vs_ref": round(pwls_high_ncc, 4),
                "nrmse_vs_ref": round(pwls_high_nrmse, 4),
            },
        ],
        "ncc_boundary": round(0.9 * pwls_low_ncc, 4),
        "nrmse_boundary": round(1.1 * pwls_low_nrmse, 4),
    }

    metrics_path = os.path.join(TASK_DIR, "evaluation", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # ---- Step 6: Visualize ----
    print("Step 6: Generating visualizations...")

    plot_reconstruction_comparison(
        phantom, recon_unweighted, recon_pwls_low,
        save_path=os.path.join(ref_dir, "reconstruction_comparison.png"),
    )
    plot_sinogram_comparison(
        sino_clean, sino_low, sino_high,
        save_path=os.path.join(ref_dir, "sinogram_comparison.png"),
    )
    plot_dose_comparison(
        phantom, recon_unweighted, recon_pwls_low, recon_pwls_high,
        save_path=os.path.join(ref_dir, "dose_comparison.png"),
    )
    plot_error_maps(
        phantom, recon_unweighted, recon_pwls_low,
        save_path=os.path.join(ref_dir, "error_maps.png"),
    )

    print("Done!")
    return metrics


if __name__ == "__main__":
    main()
