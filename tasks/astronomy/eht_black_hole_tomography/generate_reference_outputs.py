"""
Generate reference outputs for evaluation.

Runs the full BH-NeRF pipeline and saves results to evaluation/reference_outputs/.
"""

import json
import os
import numpy as np
import torch

from src.preprocessing import prepare_data
from src.solvers import BHNeRFSolver
from src.visualization import compute_metrics, compute_image_metrics

OUTPUT_DIR = "evaluation/reference_outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    obs_data, ground_truth, metadata = prepare_data("data")

    # Train BH-NeRF (use full iterations from metadata)
    print(f"Training BH-NeRF ({metadata['n_iters']} iterations)...")
    solver = BHNeRFSolver(metadata)
    result = solver.reconstruct(obs_data, seed=42)

    print(f"  Final loss: {result['loss_history'][-1]:.6f}")
    print(f"  Recovered rotation axis: {result['rot_axis']}")

    # Predict
    print("Predicting 3D emission and movie...")
    emission_3d = solver.predict_emission_3d(
        fov_M=obs_data['fov_M'],
        resolution=metadata['emission_resolution'],
    )
    pred_movie = solver.predict_movie(obs_data)

    # Evaluate
    metrics_3d = compute_metrics(emission_3d, ground_truth['emission_3d'])
    metrics_img = compute_image_metrics(pred_movie, ground_truth['images'])
    all_metrics = {**metrics_3d, **metrics_img}

    # Save
    torch.save(result['model_state_dict'], os.path.join(OUTPUT_DIR, "model_state_dict.pt"))
    np.save(os.path.join(OUTPUT_DIR, "emission_3d.npy"), emission_3d)
    np.save(os.path.join(OUTPUT_DIR, "pred_images.npy"), pred_movie)
    np.save(os.path.join(OUTPUT_DIR, "loss_history.npy"), np.array(result['loss_history']))
    np.save(os.path.join(OUTPUT_DIR, "rot_axis.npy"), result['rot_axis'])

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nReference outputs saved to {OUTPUT_DIR}/")
    for k, v in all_metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == '__main__':
    main()
