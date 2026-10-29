"""
s2ISM: Scalable Structured Illumination Microscopy via Multi-Plane ML-EM Reconstruction.

Pipeline entry point: loads data, runs reconstruction, evaluates against ground truth,
and saves metrics + visualisation.
"""

import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
from pathlib import Path

from src.solvers import max_likelihood_reconstruction
from src.visualization import plot_results, compute_metrics
from src.generate_data import generate_data

# --- Named constants (solver parameters) ---
_MAX_ITER = 50
_STOP_MODE = 'fixed'
_REP_TO_SAVE = 'last'

def main():
    task_dir = Path(__file__).parent
    data_dir = task_dir / 'data'
    eval_dir = task_dir / 'evaluation'
    ref_dir = eval_dir / 'reference_outputs'
    ref_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate data if missing ---
    needed = ['raw_data.npz', 'ground_truth.npz', 'meta_data.json']
    if not all((data_dir / f).exists() for f in needed):
        print("Data files missing — generating synthetic data...")
        generate_data(output_dir=str(data_dir))

    # --- Load data ---
    print("Loading data...")
    raw = np.load(data_dir / 'raw_data.npz')
    gt_file = np.load(data_dir / 'ground_truth.npz')

    measurements = raw['measurements'][0].astype(np.float64)   # (Ny, Nx, Nch)
    psf = raw['psf'][0].astype(np.float64)                     # (Nz, Ny, Nx, Nch)
    ground_truth = gt_file['ground_truth'][0].astype(np.float64)  # (Nz, Ny, Nx)

    with open(data_dir / 'meta_data.json') as f:
        meta = json.load(f)

    print(f"Measurements shape: {measurements.shape}")
    print(f"PSF shape:          {psf.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")

    # --- Reconstruction ---
    print(f"Running s2ISM reconstruction ({_MAX_ITER} iterations)...")
    recon, photon_counts, derivative, n_iter = max_likelihood_reconstruction(
        measurements, psf, stop=_STOP_MODE, max_iter=_MAX_ITER, rep_to_save=_REP_TO_SAVE)

    print(f"Reconstruction finished in {n_iter} iterations.")
    print(f"Reconstruction shape: {recon.shape}")

    # --- Evaluate ---
    gt_infocus = ground_truth[0]
    recon_infocus = recon[0]
    ism_sum = measurements.sum(-1)

    metrics = compute_metrics(gt_infocus, recon_infocus, ism_sum)

    print(f"\nReconstruction vs Ground Truth:")
    print(f"  NCC:   {metrics['ncc_recon']:.4f}")
    print(f"  NRMSE: {metrics['nrmse_recon']:.4f}")
    print(f"Raw ISM Sum vs Ground Truth:")
    print(f"  NCC:   {metrics['ncc_ism']:.4f}")
    print(f"  NRMSE: {metrics['nrmse_ism']:.4f}")

    # --- Save reference outputs ---
    np.savez(ref_dir / 'reconstruction.npz',
             reconstruction=recon[np.newaxis, ...].astype(np.float32))

    # --- Save metrics ---
    metrics_json = {
        'baseline': [
            {
                'method': 's2ISM ML-EM (50 iterations)',
                'ncc_vs_ref': metrics['ncc_recon'],
                'nrmse_vs_ref': metrics['nrmse_recon'],
            },
            {
                'method': 'ISM sum (confocal-like)',
                'ncc_vs_ref': metrics['ncc_ism'],
                'nrmse_vs_ref': metrics['nrmse_ism'],
            },
        ],
        'ncc_boundary': round(0.9 * metrics['ncc_recon'], 6),
        'nrmse_boundary': round(1.1 * metrics['nrmse_recon'], 6),
    }
    with open(eval_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)

    # --- Visualisation ---
    fig = plot_results(gt_infocus, ism_sum, recon_infocus,
                       save_path=ref_dir / 's2ism_result.png')
    print(f"\nResult image saved to {ref_dir / 's2ism_result.png'}")


if __name__ == '__main__':
    main()
