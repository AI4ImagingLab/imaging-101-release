"""
Ultrasound Speed-of-Sound Tomography
=====================================

Reconstructs a 2D speed-of-sound map from travel-time measurements using:
  1. Filtered Back Projection (FBP) -- baseline
  2. SART (Simultaneous Algebraic Reconstruction Technique)
  3. TV-regularized PDHG -- best quality

The reconstruction operates on slowness perturbation (delta_s = 1/c - 1/c_water),
which is naturally zero outside the object. The final speed-of-sound map is
recovered as c = 1 / (delta_s + 1/c_water).

Usage:
    cd tasks/ultrasound_sos_tomography
    python main.py
"""

import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")

TASK_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Solver constants (not in meta_data.json per CLAUDE.md) ----
_SART_PARAMS = {
    "n_iter": 30,
    "relaxation": 0.15,
}

_TV_PDHG_PARAMS = {
    "lam": 1e-6,
    "n_iter": 300,
    "positivity": False,  # slowness perturbation can be negative (fat has lower SoS than water)
}

_BACKGROUND_SOS = 1500.0  # m/s (water)


def _perturbation_to_sos(delta_s, background_sos=_BACKGROUND_SOS):
    """Convert slowness perturbation to speed of sound.

    c = 1 / (delta_s + 1/c_water)
    """
    slowness = delta_s + 1.0 / background_sos
    slowness = np.clip(slowness, 1e-8, None)  # avoid division by zero
    return 1.0 / slowness


def main():
    from src.preprocessing import load_ground_truth, load_raw_data, load_metadata
    from src.physics_model import filtered_back_projection
    from src.solvers import sart_reconstruction, tv_pdhg_reconstruction
    from src.visualization import (
        compute_ncc, compute_nrmse, compute_ssim, centre_crop,
        plot_sos_comparison, plot_sinograms, plot_loss_history,
        plot_sos_profiles,
    )

    data_dir = os.path.join(TASK_DIR, "data")
    ref_dir = os.path.join(TASK_DIR, "evaluation", "reference_outputs")
    os.makedirs(ref_dir, exist_ok=True)

    # ---- Step 1: Load data ----
    print("Step 1: Loading data...")
    gt = load_ground_truth(data_dir)
    raw = load_raw_data(data_dir)
    meta = load_metadata(data_dir)

    sos_gt = gt["sos_phantom"]
    delta_s_gt = gt["slowness_perturbation"]
    sinogram = raw["sinogram"]
    sinogram_full = raw["sinogram_full"]
    angles = raw["angles"]
    angles_full = raw["angles_full"]
    image_size = meta["image_size"]

    print(f"  SoS phantom: {sos_gt.shape}, range [{sos_gt.min():.0f}, {sos_gt.max():.0f}] m/s")
    print(f"  Slowness perturbation range: [{delta_s_gt.min():.2e}, {delta_s_gt.max():.2e}] s/m")
    print(f"  Sinogram: {sinogram.shape}, Angles: {len(angles)}")

    # ---- Step 2: FBP reconstruction (baseline) ----
    print("Step 2: FBP reconstruction...")
    delta_s_fbp = filtered_back_projection(sinogram, angles, output_size=image_size)
    sos_fbp = _perturbation_to_sos(delta_s_fbp)
    print(f"  FBP SoS range: [{sos_fbp.min():.0f}, {sos_fbp.max():.0f}] m/s")

    # ---- Step 3: SART reconstruction ----
    print(f"Step 3: SART reconstruction (n_iter={_SART_PARAMS['n_iter']})...")
    delta_s_sart, loss_sart = sart_reconstruction(
        sinogram, angles, image_size, **_SART_PARAMS
    )
    sos_sart = _perturbation_to_sos(delta_s_sart)
    print(f"  SART SoS range: [{sos_sart.min():.0f}, {sos_sart.max():.0f}] m/s")

    # ---- Step 4: TV-PDHG reconstruction ----
    print(f"Step 4: TV-PDHG reconstruction (lam={_TV_PDHG_PARAMS['lam']}, "
          f"n_iter={_TV_PDHG_PARAMS['n_iter']})...")
    delta_s_tv, loss_tv = tv_pdhg_reconstruction(
        sinogram, angles, image_size, **_TV_PDHG_PARAMS
    )
    sos_tv = _perturbation_to_sos(delta_s_tv)
    print(f"  TV-PDHG SoS range: [{sos_tv.min():.0f}, {sos_tv.max():.0f}] m/s")

    # ---- Step 5: Evaluate metrics ----
    print("Step 5: Computing metrics...")

    # Centre-crop for evaluation (avoid edge artifacts)
    crop = 0.8
    gt_crop = centre_crop(delta_s_gt, crop)
    fbp_crop = centre_crop(delta_s_fbp, crop)
    sart_crop = centre_crop(delta_s_sart, crop)
    tv_crop = centre_crop(delta_s_tv, crop)

    # Metrics on slowness perturbation (the reconstructed quantity)
    fbp_ncc = compute_ncc(fbp_crop, gt_crop)
    fbp_nrmse = compute_nrmse(fbp_crop, gt_crop)
    fbp_ssim = compute_ssim(fbp_crop, gt_crop)

    sart_ncc = compute_ncc(sart_crop, gt_crop)
    sart_nrmse = compute_nrmse(sart_crop, gt_crop)
    sart_ssim = compute_ssim(sart_crop, gt_crop)

    tv_ncc = compute_ncc(tv_crop, gt_crop)
    tv_nrmse = compute_nrmse(tv_crop, gt_crop)
    tv_ssim = compute_ssim(tv_crop, gt_crop)

    print(f"  FBP   -- NCC: {fbp_ncc:.4f}, NRMSE: {fbp_nrmse:.4f}, SSIM: {fbp_ssim:.4f}")
    print(f"  SART  -- NCC: {sart_ncc:.4f}, NRMSE: {sart_nrmse:.4f}, SSIM: {sart_ssim:.4f}")
    print(f"  TV    -- NCC: {tv_ncc:.4f}, NRMSE: {tv_nrmse:.4f}, SSIM: {tv_ssim:.4f}")

    # ---- Step 6: Save reference outputs ----
    print("Step 6: Saving reference outputs...")

    np.savez(
        os.path.join(ref_dir, "reconstructions.npz"),
        delta_s_fbp=delta_s_fbp[np.newaxis, ...],
        delta_s_sart=delta_s_sart[np.newaxis, ...],
        delta_s_tv=delta_s_tv[np.newaxis, ...],
        sos_fbp=sos_fbp[np.newaxis, ...],
        sos_sart=sos_sart[np.newaxis, ...],
        sos_tv=sos_tv[np.newaxis, ...],
        loss_history_sart=np.array(loss_sart),
        loss_history_tv=np.array(loss_tv),
    )

    # Use the best iterative method for boundary
    best_ncc = max(sart_ncc, tv_ncc)
    best_nrmse = min(sart_nrmse, tv_nrmse)

    metrics = {
        "baseline": [
            {
                "method": "FBP (60 angles)",
                "ncc_vs_ref": round(fbp_ncc, 4),
                "nrmse_vs_ref": round(fbp_nrmse, 4),
                "ssim": round(fbp_ssim, 4),
            },
            {
                "method": f"SART ({_SART_PARAMS['n_iter']} iter)",
                "ncc_vs_ref": round(sart_ncc, 4),
                "nrmse_vs_ref": round(sart_nrmse, 4),
                "ssim": round(sart_ssim, 4),
            },
            {
                "method": f"TV-PDHG (lam={_TV_PDHG_PARAMS['lam']}, {_TV_PDHG_PARAMS['n_iter']} iter)",
                "ncc_vs_ref": round(tv_ncc, 4),
                "nrmse_vs_ref": round(tv_nrmse, 4),
                "ssim": round(tv_ssim, 4),
            },
        ],
        "ncc_boundary": round(0.9 * best_ncc, 4),
        "nrmse_boundary": round(1.1 * best_nrmse, 4),
    }

    metrics_path = os.path.join(TASK_DIR, "evaluation", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Metrics saved to {metrics_path}")

    # ---- Step 7: Visualize ----
    print("Step 7: Generating visualizations...")
    plot_sos_comparison(
        sos_gt, sos_fbp, sos_tv,
        save_path=os.path.join(ref_dir, "sos_comparison.png"),
    )
    plot_sinograms(
        sinogram, sinogram_full, angles, angles_full,
        save_path=os.path.join(ref_dir, "sinograms.png"),
    )
    plot_loss_history(
        loss_sart, loss_tv,
        save_path=os.path.join(ref_dir, "convergence.png"),
    )
    plot_sos_profiles(
        sos_gt, sos_fbp, sos_tv,
        save_path=os.path.join(ref_dir, "sos_profiles.png"),
    )

    print("Done!")
    return metrics


if __name__ == "__main__":
    main()
