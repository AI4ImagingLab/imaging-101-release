"""
Non-Cartesian MRI Reconstruction with L1-Wavelet Compressed Sensing
====================================================================

Pipeline:
    1. Load and preprocess non-Cartesian (radial) multi-coil MRI data
    2. Run gridding reconstruction (baseline)
    3. Run L1-wavelet CS reconstruction via SigPy NUFFT
    4. Evaluate reconstruction quality (NRMSE, NCC, PSNR)
    5. Visualize results and save outputs
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.preprocessing import prepare_data
from src.physics_model import gridding_reconstruct, compute_density_compensation
from src.solvers import l1wav_reconstruct_single
from src.visualization import (
    compute_metrics,
    plot_reconstruction_comparison,
    plot_error_maps,
    plot_trajectory,
    print_metrics_table,
)

# ── Solver parameters (not in meta_data.json to avoid leaking algorithm info) ──
_L1WAV_LAMBDA = 5e-5
_L1WAV_MAX_ITER = 100
_DCF_MAX_ITER = 30


def main(data_dir: str = "data", output_dir: str = "evaluation/reference_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load data ──
    print("Step 1: Loading data...")
    obs_data, ground_truth, metadata = prepare_data(data_dir)
    kdata = obs_data["kdata"][0].astype(np.complex128)       # (C, M)
    coord = obs_data["coord"][0].astype(np.float64)           # (M, 2)
    coil_maps = obs_data["coil_maps"][0].astype(np.complex128)  # (C, H, W)
    phantom = ground_truth[0].astype(np.complex128)           # (H, W)
    gt_mag = np.abs(phantom)

    image_shape = coil_maps.shape[1:]
    n_coils = coil_maps.shape[0]
    n_pts = coord.shape[0]
    print(f"  {n_coils} coils, {n_pts} k-space points, "
          f"{image_shape[0]}x{image_shape[1]} image")

    # ── Step 2: Density compensation ──
    print(f"\nStep 2: Computing density compensation ({_DCF_MAX_ITER} iterations)...")
    dcf = compute_density_compensation(coord, image_shape, max_iter=_DCF_MAX_ITER)

    # ── Step 3: Gridding reconstruction (baseline) ──
    print("\nStep 3: Gridding reconstruction...")
    gridding_recon = gridding_reconstruct(kdata, coord, coil_maps, dcf)
    gridding_mag = np.abs(gridding_recon)

    # ── Step 4: L1-Wavelet CS reconstruction ──
    print(f"\nStep 4: L1-Wavelet CS reconstruction "
          f"(lambda={_L1WAV_LAMBDA}, max_iter={_L1WAV_MAX_ITER})...")
    l1wav_recon = l1wav_reconstruct_single(
        kdata, coord, coil_maps, lamda=_L1WAV_LAMBDA, max_iter=_L1WAV_MAX_ITER
    )
    l1wav_mag = np.abs(l1wav_recon)

    # Save reconstructions
    np.savez_compressed(
        os.path.join(output_dir, "gridding_reconstruction.npz"),
        reconstruction=gridding_recon[np.newaxis].astype(np.complex64),
        dcf=dcf[np.newaxis].astype(np.float32),
    )
    np.savez_compressed(
        os.path.join(output_dir, "l1wav_reconstruction.npz"),
        reconstruction=l1wav_recon[np.newaxis].astype(np.complex64),
    )

    # ── Step 5: Evaluate ──
    print("\nStep 5: Evaluating reconstruction quality...")
    gridding_metrics = compute_metrics(gridding_mag, gt_mag)
    l1wav_metrics = compute_metrics(l1wav_mag, gt_mag)

    all_metrics = {
        "Gridding": gridding_metrics,
        "L1-Wavelet CS": l1wav_metrics,
    }
    print_metrics_table(all_metrics)

    # Save metrics
    detail = {
        "gridding": gridding_metrics,
        "l1wav": l1wav_metrics,
        "lambda": _L1WAV_LAMBDA,
        "max_iter": _L1WAV_MAX_ITER,
    }
    with open(os.path.join(output_dir, "metrics_detail.json"), "w") as f:
        json.dump(detail, f, indent=2)

    # ── Step 6: Visualize ──
    print("\nStep 6: Generating visualizations...")
    plot_reconstruction_comparison(
        gt_mag, gridding_mag, l1wav_mag,
        save_path=os.path.join(output_dir, "reconstruction_comparison.png"),
    )
    plot_error_maps(
        gt_mag, gridding_mag, l1wav_mag,
        save_path=os.path.join(output_dir, "error_maps.png"),
    )
    n_spokes = metadata.get("n_spokes", None)
    plot_trajectory(
        coord, n_spokes=n_spokes,
        save_path=os.path.join(output_dir, "radial_trajectory.png"),
    )
    print("  Saved figures to", output_dir)

    print("\nDone!")
    return l1wav_recon, all_metrics


if __name__ == "__main__":
    main()
