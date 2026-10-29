"""
PET MLEM Reconstruction
========================

Reconstructs a 2D PET activity distribution from Poisson-noisy
sinogram data using Maximum Likelihood Expectation Maximization (MLEM)
and Ordered Subsets EM (OSEM).

Pipeline:
1. Load PET sinogram data, background estimate, and metadata
2. MLEM reconstruction
3. OSEM reconstruction (accelerated)
4. Evaluate against ground truth
5. Save results and visualizations

Usage:
    cd tasks/pet_mlem
    python main.py
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import (
    load_sinogram_data,
    load_ground_truth,
    load_metadata,
    preprocess_sinogram,
)
from src.solvers import solve_mlem, solve_osem
from src.visualization import (
    compute_ncc,
    compute_nrmse,
    plot_pet_reconstruction,
    plot_convergence,
    plot_sinogram,
)

# Solver parameters
_MLEM_PARAMS = {'n_iter': 50}
_OSEM_PARAMS = {'n_iter': 10, 'n_subsets': 6}


def main():
    print("=" * 60)
    print("PET MLEM Reconstruction")
    print("=" * 60)

    # --- Check if data exists, generate if needed ---
    data_dir = os.path.join(TASK_DIR, 'data')
    if not os.path.exists(os.path.join(data_dir, 'raw_data.npz')):
        print("\nData not found. Generating synthetic data...")
        from src.generate_data import generate_synthetic_data, save_data
        data = generate_synthetic_data()
        save_data(data, TASK_DIR)
        print("Data generation complete.")

    # --- Step 1: Load data ---
    print("\n[1/5] Loading PET sinogram data...")
    sinogram, background, theta = load_sinogram_data(TASK_DIR)
    activity_gt = load_ground_truth(TASK_DIR)
    meta = load_metadata(TASK_DIR)
    N = meta['image_size']

    print(f"  Image size: {N}x{N}")
    print(f"  Sinogram shape: {sinogram.shape}")
    print(f"  Angles: {meta['n_angles']}")
    print(f"  Count level: {meta['count_level']}")
    print(f"  Noise model: {meta['noise_model']}")

    # Preprocess
    sino_2d = preprocess_sinogram(sinogram)
    bg_2d = preprocess_sinogram(background)
    theta = theta.squeeze(0)  # remove batch dim for solver
    gt_2d = activity_gt[0]

    # --- Step 2: MLEM reconstruction ---
    print(f"\n[2/5] MLEM reconstruction ({_MLEM_PARAMS['n_iter']} iterations)...")
    recon_mlem, ll_mlem = solve_mlem(
        sino_2d, theta, N,
        n_iter=_MLEM_PARAMS['n_iter'],
        background=bg_2d,
    )

    # --- Step 3: OSEM reconstruction ---
    print(f"\n[3/5] OSEM reconstruction ({_OSEM_PARAMS['n_iter']} iterations, "
          f"{_OSEM_PARAMS['n_subsets']} subsets)...")
    recon_osem, ll_osem = solve_osem(
        sino_2d, theta, N,
        n_iter=_OSEM_PARAMS['n_iter'],
        n_subsets=_OSEM_PARAMS['n_subsets'],
        background=bg_2d,
    )

    # --- Step 4: Evaluate ---
    print("\n[4/5] Evaluating results...")

    # Mask for evaluation (inside FOV where activity > 0)
    mask = gt_2d > 0

    ncc_mlem = compute_ncc(recon_mlem, gt_2d, mask=mask)
    nrmse_mlem = compute_nrmse(recon_mlem, gt_2d, mask=mask)
    print(f"  MLEM:  NCC={ncc_mlem:.4f}, NRMSE={nrmse_mlem:.4f}")

    ncc_osem = compute_ncc(recon_osem, gt_2d, mask=mask)
    nrmse_osem = compute_nrmse(recon_osem, gt_2d, mask=mask)
    print(f"  OSEM:  NCC={ncc_osem:.4f}, NRMSE={nrmse_osem:.4f}")

    # --- Step 5: Save results ---
    print("\n[5/5] Saving results...")

    output_dir = os.path.join(TASK_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)
    ref_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
    os.makedirs(ref_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(ref_dir, 'recon_mlem.npz'),
        reconstruction=recon_mlem[np.newaxis, ...].astype(np.float32),
        log_likelihood=np.array(ll_mlem, dtype=np.float32),
    )
    np.savez_compressed(
        os.path.join(ref_dir, 'recon_osem.npz'),
        reconstruction=recon_osem[np.newaxis, ...].astype(np.float32),
        log_likelihood=np.array(ll_osem, dtype=np.float32),
    )

    # Use best (OSEM) as boundary
    best_ncc = max(ncc_mlem, ncc_osem)
    best_nrmse = min(nrmse_mlem, nrmse_osem)

    metrics = {
        "baseline": [
            {
                "method": f"MLEM ({_MLEM_PARAMS['n_iter']} iterations)",
                "ncc_vs_ref": round(ncc_mlem, 4),
                "nrmse_vs_ref": round(nrmse_mlem, 4),
            },
            {
                "method": f"OSEM ({_OSEM_PARAMS['n_iter']} iters, {_OSEM_PARAMS['n_subsets']} subsets)",
                "ncc_vs_ref": round(ncc_osem, 4),
                "nrmse_vs_ref": round(nrmse_osem, 4),
            },
        ],
        "ncc_boundary": round(0.9 * best_ncc, 4),
        "nrmse_boundary": round(1.1 * best_nrmse, 4),
    }
    metrics_path = os.path.join(TASK_DIR, 'evaluation', 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Visualizations
    plot_sinogram(sino_2d, title="Noisy PET Sinogram",
                  save_path=os.path.join(output_dir, 'sinogram.png'))
    plot_pet_reconstruction(gt_2d, recon_mlem, recon_osem,
                            save_path=os.path.join(output_dir, 'reconstructions.png'))
    plot_convergence(ll_mlem, ll_osem,
                     save_path=os.path.join(output_dir, 'convergence.png'))

    print(f"  Figures saved to {output_dir}/")
    print("\nDone!")

    return metrics


if __name__ == '__main__':
    main()
