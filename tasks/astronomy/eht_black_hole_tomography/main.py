"""
EHT Black Hole Tomography (BH-NeRF)
====================================

Reconstruct 3D black hole emission from time-series interferometric
measurements using a neural radiance field with Keplerian orbital dynamics.

Reference: Levis et al. "Gravitationally Lensed Black Hole Emission
Tomography" (CVPR 2022)

Usage:
    python main.py
"""

import json
import os
import numpy as np

from src.preprocessing import prepare_data
from src.solvers import BHNeRFSolver
from src.visualization import (
    compute_metrics,
    compute_image_metrics,
    plot_emission_slices,
    plot_movie_comparison,
    plot_lightcurve,
    plot_loss_curves,
    print_metrics_table,
)


def main():
    # ----------------------------------------------------------------
    # Step 1: Load and preprocess data
    # ----------------------------------------------------------------
    print("Loading data...")
    obs_data, ground_truth, metadata = prepare_data("data")

    print(f"  Image frames: {len(obs_data['t_frames'])}")
    print(f"  Image size: {obs_data['images'].shape[1:]}")
    print(f"  Emission resolution: {ground_truth['emission_3d'].shape}")

    # ----------------------------------------------------------------
    # Step 2: Train BH-NeRF
    # ----------------------------------------------------------------
    print("\nTraining BH-NeRF model...")
    solver = BHNeRFSolver(metadata)
    result = solver.reconstruct(obs_data)

    print(f"\n  Final loss: {result['loss_history'][-1]:.6f}")
    print(f"  Recovered rotation axis: {result['rot_axis']}")
    print(f"  True rotation axis:      {ground_truth['rot_axis']}")

    # ----------------------------------------------------------------
    # Step 3: Predict 3D emission and movie
    # ----------------------------------------------------------------
    print("\nPredicting 3D emission and movie...")
    emission_3d = solver.predict_emission_3d(
        fov_M=obs_data['fov_M'],
        resolution=metadata['emission_resolution'],
    )
    pred_movie = solver.predict_movie(obs_data)

    # ----------------------------------------------------------------
    # Step 4: Evaluate
    # ----------------------------------------------------------------
    print("\nEvaluating results...")
    metrics_3d = compute_metrics(emission_3d, ground_truth['emission_3d'])
    metrics_img = compute_image_metrics(pred_movie, ground_truth['images'])

    all_metrics = {**metrics_3d, **metrics_img}
    print_metrics_table(all_metrics)

    # ----------------------------------------------------------------
    # Step 5: Visualize and save
    # ----------------------------------------------------------------
    os.makedirs("evaluation/reference_outputs", exist_ok=True)

    plot_emission_slices(emission_3d, obs_data['fov_M'],
                         ground_truth=ground_truth['emission_3d'],
                         save_path="evaluation/reference_outputs/emission_slices.png")

    plot_movie_comparison(pred_movie, ground_truth['images'],
                          obs_data['t_frames'],
                          save_path="evaluation/reference_outputs/movie_comparison.png")

    plot_lightcurve(pred_movie, ground_truth['images'],
                    obs_data['t_frames'],
                    save_path="evaluation/reference_outputs/lightcurve.png")

    plot_loss_curves(result['loss_history'],
                     save_path="evaluation/reference_outputs/loss_curve.png")

    # Save primary output
    np.save("evaluation/reference_outputs/reconstruction.npy", emission_3d)
    np.save("evaluation/reference_outputs/pred_images.npy", pred_movie)
    np.save("evaluation/reference_outputs/loss_history.npy", np.array(result['loss_history']))

    with open("evaluation/reference_outputs/metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nResults saved to evaluation/reference_outputs/")
    print("Done.")


if __name__ == '__main__':
    main()
