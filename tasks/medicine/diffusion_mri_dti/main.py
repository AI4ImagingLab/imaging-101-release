"""
Diffusion MRI DTI: Diffusion Tensor Estimation
===============================================

Estimates diffusion tensor maps from synthetic multi-direction
diffusion-weighted MRI data using the Stejskal-Tanner equation.

Pipeline:
1. Load DWI data, gradient table, and metadata
2. Preprocess: extract S0, ensure non-negative
3. Fit tensors using OLS and WLS methods
4. Eigendecompose tensors to get FA/MD maps
5. Evaluate against ground truth
6. Save results and visualizations

Usage:
    cd tasks/diffusion_mri_dti
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
    load_dwi_data,
    load_ground_truth,
    load_metadata,
    preprocess_dwi,
)
from src.solvers import fit_dti_ols, fit_dti_wls, tensor_eig_decomposition
from src.visualization import compute_ncc, compute_nrmse, plot_dti_maps, plot_color_fa


def main():
    print("=" * 60)
    print("Diffusion MRI DTI: Diffusion Tensor Estimation")
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
    print("\n[1/6] Loading diffusion MRI data...")
    dwi_signal, bvals, bvecs = load_dwi_data(TASK_DIR)
    fa_gt, md_gt, tensor_gt, tissue_mask = load_ground_truth(TASK_DIR)
    meta = load_metadata(TASK_DIR)

    print(f"  DWI signal shape: {dwi_signal.shape}")
    print(f"  b-values: {np.unique(bvals)} s/mm^2")
    print(f"  Gradient directions: {(bvals > 10).sum()}")
    print(f"  Tissue pixels: {tissue_mask.sum()}")

    # --- Step 2: Preprocess ---
    print("\n[2/6] Preprocessing...")
    dwi_2d, S0_2d = preprocess_dwi(dwi_signal, bvals, bvecs)
    fa_gt_2d = fa_gt[0]
    md_gt_2d = md_gt[0]
    mask_2d = tissue_mask[0]

    # --- Step 3: Fit tensors ---
    print("\n[3/6] Fitting diffusion tensors...")

    # OLS fit (fast baseline)
    print("  OLS fit...")
    tensor_ols, S0_ols = fit_dti_ols(dwi_2d, bvals, bvecs, mask=mask_2d)

    # WLS fit (improved)
    print("  WLS fit...")
    tensor_wls, S0_wls = fit_dti_wls(dwi_2d, bvals, bvecs, mask=mask_2d)

    # --- Step 4: Eigendecomposition ---
    print("\n[4/6] Computing FA/MD maps...")
    evals_ols, evecs_ols, fa_ols, md_ols = tensor_eig_decomposition(tensor_ols, mask=mask_2d)
    evals_wls, evecs_wls, fa_wls, md_wls = tensor_eig_decomposition(tensor_wls, mask=mask_2d)

    # --- Step 5: Evaluate ---
    print("\n[5/6] Evaluating results...")

    # OLS metrics
    ncc_fa_ols = compute_ncc(fa_ols, fa_gt_2d, mask=mask_2d)
    nrmse_fa_ols = compute_nrmse(fa_ols, fa_gt_2d, mask=mask_2d)
    ncc_md_ols = compute_ncc(md_ols, md_gt_2d, mask=mask_2d)
    nrmse_md_ols = compute_nrmse(md_ols, md_gt_2d, mask=mask_2d)
    print(f"  OLS FA:  NCC={ncc_fa_ols:.4f}, NRMSE={nrmse_fa_ols:.4f}")
    print(f"  OLS MD:  NCC={ncc_md_ols:.4f}, NRMSE={nrmse_md_ols:.4f}")

    # WLS metrics
    ncc_fa_wls = compute_ncc(fa_wls, fa_gt_2d, mask=mask_2d)
    nrmse_fa_wls = compute_nrmse(fa_wls, fa_gt_2d, mask=mask_2d)
    ncc_md_wls = compute_ncc(md_wls, md_gt_2d, mask=mask_2d)
    nrmse_md_wls = compute_nrmse(md_wls, md_gt_2d, mask=mask_2d)
    print(f"  WLS FA:  NCC={ncc_fa_wls:.4f}, NRMSE={nrmse_fa_wls:.4f}")
    print(f"  WLS MD:  NCC={ncc_md_wls:.4f}, NRMSE={nrmse_md_wls:.4f}")

    # --- Step 6: Save results ---
    print("\n[6/6] Saving results...")

    output_dir = os.path.join(TASK_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)
    ref_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
    os.makedirs(ref_dir, exist_ok=True)

    # Save reference outputs (WLS as primary)
    np.savez_compressed(
        os.path.join(ref_dir, 'dti_ols.npz'),
        fa_map=fa_ols[np.newaxis, ...].astype(np.float32),
        md_map=md_ols[np.newaxis, ...].astype(np.float32),
        tensor_elements=tensor_ols[np.newaxis, ...].astype(np.float32),
    )
    np.savez_compressed(
        os.path.join(ref_dir, 'dti_wls.npz'),
        fa_map=fa_wls[np.newaxis, ...].astype(np.float32),
        md_map=md_wls[np.newaxis, ...].astype(np.float32),
        tensor_elements=tensor_wls[np.newaxis, ...].astype(np.float32),
    )

    # Save metrics (evaluate FA map as primary output)
    metrics = {
        "baseline": [
            {
                "method": "OLS tensor fit",
                "ncc_vs_ref": round(ncc_fa_ols, 4),
                "nrmse_vs_ref": round(nrmse_fa_ols, 4),
                "ncc_md": round(ncc_md_ols, 4),
                "nrmse_md": round(nrmse_md_ols, 4),
            },
            {
                "method": "WLS tensor fit",
                "ncc_vs_ref": round(ncc_fa_wls, 4),
                "nrmse_vs_ref": round(nrmse_fa_wls, 4),
                "ncc_md": round(ncc_md_wls, 4),
                "nrmse_md": round(nrmse_md_wls, 4),
            },
        ],
        "ncc_boundary": round(0.9 * ncc_fa_wls, 4),
        "nrmse_boundary": round(1.1 * nrmse_fa_wls, 4),
    }
    metrics_path = os.path.join(TASK_DIR, 'evaluation', 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Visualizations
    plot_dti_maps(
        fa_gt_2d, fa_ols, md_gt_2d, md_ols, mask_2d,
        title_est="OLS",
        save_path=os.path.join(output_dir, 'dti_ols.png'),
    )
    plot_dti_maps(
        fa_gt_2d, fa_wls, md_gt_2d, md_wls, mask_2d,
        title_est="WLS",
        save_path=os.path.join(output_dir, 'dti_wls.png'),
    )
    plot_color_fa(
        fa_wls, evecs_wls, mask_2d,
        save_path=os.path.join(output_dir, 'color_fa_wls.png'),
    )

    print(f"\n  Figures saved to {output_dir}/")
    print("\nDone!")

    return metrics


if __name__ == '__main__':
    main()
