"""
EHT Black Hole Probabilistic Imaging (DPI)
============================================

Main script orchestrating the DPI reconstruction pipeline:
  1. Load and preprocess EHT observation data
  2. Train normalizing flow to learn posterior distribution
  3. Sample posterior and compute statistics
  4. Evaluate reconstruction quality and uncertainty
  5. Visualize and save results

Usage
-----
    cd tasks/eht_black_hole_UQ
    python main.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')

from src.preprocessing import prepare_data, load_ground_truthfrom src.solvers import DPISolver
from src.visualization import (
    plot_posterior_summary,
    plot_posterior_samples,
    plot_loss_curves,
    compute_metrics,
    compute_uq_metrics,
    print_metrics_table,
)


def main():
    # ── Step 1: Load and preprocess observation data ──────────────────────
    print("Step 1: Loading and preprocessing observation data...")
    (obs_data, closure_indices, nufft_params,
     prior_image, flux_const, metadata) = prepare_data("data")

    npix = metadata["npix"]
    fov_uas = metadata["fov_uas"]

    print(f"  Image size      : {npix}x{npix}")
    print(f"  FOV             : {fov_uas} uas")
    print(f"  Visibilities    : {len(obs_data['vis'])}")
    print(f"  Closure phases  : {len(closure_indices['cphase_data']['cphase'])}")
    print(f"  Closure amps    : {len(closure_indices['camp_data']['camp'])}")
    print(f"  Flux estimate   : {flux_const:.4f} Jy")

    # ── Step 2: Train DPI model ──────────────────────────────────────────
    print("\nStep 2: Training DPI normalizing flow...")
    solver = DPISolver(
        npix=npix,
        n_flow=metadata["n_flow"],
        seqfrac=metadata.get("seqfrac", 4),
        n_epoch=metadata["n_epoch"],
        batch_size=metadata["batch_size"],
        lr=metadata["lr"],
        logdet_weight=metadata["logdet_weight"],
        l1_weight=metadata["l1_weight"],
        tsv_weight=metadata["tsv_weight"],
        flux_weight=metadata["flux_weight"],
        center_weight=metadata["center_weight"],
        mem_weight=metadata["mem_weight"],
        grad_clip=metadata["grad_clip"],
    )

    result = solver.reconstruct(
        obs_data, closure_indices, nufft_params, prior_image, flux_const
    )

    # ── Step 3: Sample posterior ─────────────────────────────────────────
    print("\nStep 3: Sampling posterior distribution...")
    posterior = solver.posterior_statistics(n_samples=1000)
    print(f"  Posterior mean peak : {posterior['mean'].max():.6f}")
    print(f"  Posterior mean std  : {posterior['std'].mean():.6f}")

    # ── Step 4: Evaluate ─────────────────────────────────────────────────
    print("\nStep 4: Evaluating reconstruction quality...")
    ground_truth = load_ground_truth("data")

    metrics = compute_metrics(posterior['mean'], ground_truth)
    uq_metrics = compute_uq_metrics(posterior['mean'], posterior['std'],
                                      ground_truth)
    print()
    print_metrics_table(uq_metrics)

    # ── Step 5: Visualize and save ───────────────────────────────────────
    print("\nStep 5: Generating visualizations and saving outputs...")
    os.makedirs("evaluation/reference_outputs", exist_ok=True)

    plot_posterior_summary(
        posterior['mean'], posterior['std'], posterior['samples'],
        ground_truth=ground_truth,
        pixel_size_uas=fov_uas / npix,
        save_path="evaluation/reference_outputs/posterior_summary.png"
    )
    plot_posterior_samples(
        posterior['samples'], n_show=8,
        pixel_size_uas=fov_uas / npix,
        save_path="evaluation/reference_outputs/posterior_samples.png"
    )
    plot_loss_curves(result['loss_history'],
                     save_path="evaluation/reference_outputs/loss_curves.png")

    np.save("evaluation/reference_outputs/posterior_mean.npy", posterior['mean'])
    np.save("evaluation/reference_outputs/posterior_std.npy", posterior['std'])
    np.save("evaluation/reference_outputs/posterior_samples.npy",
            posterior['samples'][:100])
    np.save("evaluation/reference_outputs/ground_truth.npy", ground_truth)

    # Save metrics
    with open("evaluation/reference_outputs/metrics.json", "w") as f:
        json.dump(uq_metrics, f, indent=2)

    print("\nDone. Outputs saved to evaluation/reference_outputs/")
    return posterior, uq_metrics


if __name__ == "__main__":
    main()
