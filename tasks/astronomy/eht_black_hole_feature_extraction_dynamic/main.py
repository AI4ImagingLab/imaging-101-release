"""
Dynamic EHT Black Hole Feature Extraction (α-DPI)
===================================================

Main script for per-frame geometric parameter inference from time-varying
EHT closure quantities using α-DPI.

Pipeline:
  1. Load metadata and per-frame observations
  2. For each time frame: train α-DPI, importance resample
  3. Collect posteriors and generate ridge plots

Usage
-----
    cd tasks/eht_black_hole_feature_extraction_dynamic
    python main.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')

from src.preprocessing import prepare_frame, load_metadata, load_raw_data
from src.solvers import AlphaDPISolver
from src.visualization import (
    plot_ridge,
    plot_param_evolution,
    plot_frame_images,
    plot_loss_curves,
    compute_frame_metrics,
    print_frame_metrics,
)


def main():
    # ── Step 1: Load metadata ───────────────────────────────────────────────
    print("Step 1: Loading metadata...")
    metadata = load_metadata("data")
    n_frames = metadata["n_frames"]
    npix = metadata["npix"]
    fov_uas = metadata["fov_uas"]
    frame_times = np.array(metadata["frame_times_hr"])

    # Load ground truth from ground_truth.npz
    import numpy as _np
    _gt = _np.load("data/ground_truth.npz")
    _pa   = _gt["position_angle_deg"]
    _diam = _gt["diameter_uas"]
    _wid  = _gt["width_uas"]
    _asym = _gt["asymmetry"]
    gt_per_frame = [
        {"position_angle_deg": float(_pa[i]),
         "diameter_uas":       float(_diam[i]),
         "width_uas":          float(_wid[i]),
         "asymmetry":          float(_asym[i])}
        for i in range(n_frames)
    ]

    print(f"  Frames         : {n_frames}")
    print(f"  Image          : {npix}x{npix}, FOV={fov_uas} μas")
    print(f"  Duration       : {metadata['obs_duration_hr']} hr")
    print(f"  Geometric model: {metadata['geometric_model']}")

    # ── Step 2: Per-frame α-DPI inference ───────────────────────────────────
    print("\nStep 2: Per-frame α-DPI inference...")
    raw_data = load_raw_data("data")

    all_params = []
    all_weights = []
    all_images = []
    all_losses = []

    for i in range(n_frames):
        print(f"\n{'=' * 60}")
        print(f"Frame {i}/{n_frames - 1}  "
              f"(t = {frame_times[i]:.2f} hr, "
              f"PA_gt = {gt_per_frame[i]['position_angle_deg']:.1f} deg)")
        print(f"{'=' * 60}")

        # Load and preprocess this frame
        obs_data, closure_indices, nufft_params, flux_const = prepare_frame(
            raw_data, i, npix, fov_uas)

        n_cp = len(closure_indices['cphase_data']['cphase'])
        n_ca = len(closure_indices['camp_data']['camp'])
        print(f"  Vis: {len(obs_data['vis'])}, "
              f"CP: {n_cp}, CA: {n_ca}, Flux: {flux_const:.3f} Jy")

        if n_cp < 3 or n_ca < 3:
            print(f"  WARNING: Too few closure quantities, skipping frame {i}")
            # Use placeholder values
            all_params.append(np.zeros((100, 4)))
            all_weights.append(np.ones(100) / 100)
            all_images.append(np.zeros((npix, npix)))
            all_losses.append({})
            continue

        # Initialize solver
        solver = AlphaDPISolver(
            npix=npix, fov_uas=fov_uas,
            n_flow=metadata["n_flow"],
            seqfrac=1.0 / metadata.get("seqfrac_inv", 4),
            n_epoch=metadata["n_epoch"],
            batch_size=metadata["batch_size"],
            lr=metadata["lr"],
            logdet_weight=metadata["logdet_weight"],
            grad_clip=metadata["grad_clip"],
            alpha=metadata.get("alpha_divergence", 1.0),
            beta=metadata.get("beta", 0.0),
            start_order=metadata.get("start_order", 4),
            decay_rate=metadata.get("decay_rate", 1000),
            geometric_model=metadata.get("geometric_model", "simple_crescent"),
            n_gaussian=metadata.get("n_gaussian", 0),
            r_range=metadata.get("r_range", [10.0, 40.0]),
            width_range=metadata.get("width_range", [1.0, 40.0]),
        )

        # Train
        result = solver.reconstruct(
            obs_data, closure_indices, nufft_params, flux_const)
        all_losses.append(result['loss_history'])

        # Importance sampling
        posterior = solver.importance_resample(
            obs_data, closure_indices, nufft_params, n_samples=10000)

        all_params.append(posterior['params_physical'])
        all_weights.append(posterior['importance_weights'])
        all_images.append(posterior['weighted_mean_image'])

        # Print frame summary
        w = posterior['importance_weights']
        w = w / w.sum()
        n_eff = 1.0 / np.sum(w ** 2)
        means = np.sum(w[:, None] * posterior['params_physical'], axis=0)
        gt = gt_per_frame[i]
        print(f"  N_eff: {n_eff:.0f}")
        print(f"  Diameter: {means[0]:.1f} (gt {gt['diameter_uas']:.1f}), "
              f"Width: {means[1]:.1f} (gt {gt['width_uas']:.1f})")
        print(f"  Asym: {means[2]:.2f} (gt {gt['asymmetry']:.2f}), "
              f"PA: {means[3]:.1f} (gt {gt['position_angle_deg']:.1f})")

    # ── Step 3: Generate visualizations ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 3: Generating visualizations...")
    print(f"{'=' * 60}")
    os.makedirs("evaluation/reference_outputs", exist_ok=True)

    param_names = ["diameter (μas)", "width (μas)", "asymmetry", "PA (deg)"]

    # Ridge plot (Figure 13 style)
    plot_ridge(
        all_params, param_names, gt_per_frame, all_weights,
        frame_times, save_path="evaluation/reference_outputs/ridge_plot.png")

    # Parameter evolution with error bars
    plot_param_evolution(
        all_params, param_names, gt_per_frame, all_weights,
        frame_times, save_path="evaluation/reference_outputs/param_evolution.png")

    # Frame images montage
    # Generate ground truth images for comparison
    from src.generate_data import generate_simple_crescent_image
    gt_images = []
    for gt in gt_per_frame:
        gt_img = generate_simple_crescent_image(
            npix, fov_uas,
            gt['diameter_uas'], gt['width_uas'],
            gt['asymmetry'], gt['position_angle_deg'])
        gt_images.append(gt_img)

    plot_frame_images(
        all_images, frame_times,
        pixel_size_uas=fov_uas / npix,
        gt_images=gt_images,
        save_path="evaluation/reference_outputs/frame_images.png")

    # ── Step 4: Compute and print metrics ───────────────────────────────────
    print("\nStep 4: Parameter recovery metrics...")
    metrics = compute_frame_metrics(
        all_params, gt_per_frame, all_weights, param_names)
    print_frame_metrics(metrics)

    # ── Step 5: Save outputs ────────────────────────────────────────────────
    print("\nStep 5: Saving outputs...")
    np.save("evaluation/reference_outputs/all_params.npy", np.array(all_params))
    np.save("evaluation/reference_outputs/all_weights.npy", np.array(all_weights))
    np.save("evaluation/reference_outputs/all_images.npy", np.array(all_images))

    # Save per-frame loss histories
    for i, loss in enumerate(all_losses):
        if loss:
            np.save(f"evaluation/reference_outputs/loss_frame_{i:02d}.npy", loss)

    # Save metrics
    all_metrics = {
        'n_frames': n_frames,
        'frame_times_hr': frame_times.tolist(),
        'ground_truth_per_frame': gt_per_frame,
        'posterior_means': metrics['means'].tolist(),
        'posterior_stds': metrics['stds'].tolist(),
        'biases': metrics['biases'].tolist(),
        'avg_abs_bias': np.mean(np.abs(metrics['biases']), axis=0).tolist(),
        'avg_std': np.mean(metrics['stds'], axis=0).tolist(),
    }
    with open("evaluation/reference_outputs/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\nDone. Outputs saved to evaluation/reference_outputs/")
    return all_params, all_weights, all_metrics


if __name__ == "__main__":
    main()
