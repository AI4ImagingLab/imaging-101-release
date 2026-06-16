"""
MRI T2 Mapping: Quantitative Parameter Estimation
==================================================

Estimates T2 relaxation time maps from synthetic multi-echo spin-echo
MRI data using mono-exponential signal model fitting.

Pipeline:
1. Load multi-echo MRI data and metadata
2. Fit T2 maps using log-linear and nonlinear least-squares methods
3. Evaluate against ground truth
4. Save results and visualizations

Usage:
    cd tasks/mri_t2_mapping
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
    load_multi_echo_data,
    load_ground_truth,
    load_metadata,
    preprocess_signal,
)
from src.solvers import fit_t2_loglinear, fit_t2_nonlinear
from src.visualization import compute_ncc, compute_nrmse, plot_t2_maps, plot_signal_decay


def main():
    print("=" * 60)
    print("MRI T2 Mapping: Quantitative Parameter Estimation")
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
    print("\n[1/5] Loading multi-echo MRI data...")
    signal = load_multi_echo_data(TASK_DIR)
    T2_gt, M0_gt, tissue_mask = load_ground_truth(TASK_DIR)
    meta = load_metadata(TASK_DIR)
    TE = meta['echo_times_ms']

    print(f"  Signal shape: {signal.shape}")
    print(f"  Echo times: {TE} ms")
    print(f"  Tissue pixels: {tissue_mask.sum()}")

    # --- Step 2: Preprocess ---
    print("\n[2/5] Preprocessing...")
    signal_2d = preprocess_signal(signal)
    T2_gt_2d = T2_gt[0]
    M0_gt_2d = M0_gt[0]
    mask_2d = tissue_mask[0]

    # --- Step 3: Fit T2 maps ---
    print("\n[3/5] Fitting T2 maps...")

    # Log-linear fit (fast baseline)
    print("  Log-linear fit...")
    T2_ll, M0_ll = fit_t2_loglinear(signal_2d, TE, mask=mask_2d)

    # Nonlinear least-squares fit
    print("  Nonlinear least-squares fit...")
    T2_nls, M0_nls = fit_t2_nonlinear(
        signal_2d, TE, mask=mask_2d,
        T2_init=T2_ll, M0_init=M0_ll,
    )

    # --- Step 4: Evaluate ---
    print("\n[4/5] Evaluating results...")

    # Log-linear metrics
    ncc_ll = compute_ncc(T2_ll, T2_gt_2d, mask=mask_2d)
    nrmse_ll = compute_nrmse(T2_ll, T2_gt_2d, mask=mask_2d)
    print(f"  Log-linear:  NCC={ncc_ll:.4f}, NRMSE={nrmse_ll:.4f}")

    # Nonlinear LS metrics
    ncc_nls = compute_ncc(T2_nls, T2_gt_2d, mask=mask_2d)
    nrmse_nls = compute_nrmse(T2_nls, T2_gt_2d, mask=mask_2d)
    print(f"  Nonlinear LS: NCC={ncc_nls:.4f}, NRMSE={nrmse_nls:.4f}")

    # --- Step 5: Save results ---
    print("\n[5/5] Saving results...")

    output_dir = os.path.join(TASK_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)
    ref_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
    os.makedirs(ref_dir, exist_ok=True)

    # Save reference outputs
    np.savez_compressed(
        os.path.join(ref_dir, 'T2_map_loglinear.npz'),
        T2_map=T2_ll[np.newaxis, ...].astype(np.float32),
        M0_map=M0_ll[np.newaxis, ...].astype(np.float32),
    )
    np.savez_compressed(
        os.path.join(ref_dir, 'T2_map_nonlinear.npz'),
        T2_map=T2_nls[np.newaxis, ...].astype(np.float32),
        M0_map=M0_nls[np.newaxis, ...].astype(np.float32),
    )

    # Save metrics
    metrics = {
        "baseline": [
            {
                "method": "Log-linear T2 fit",
                "ncc_vs_ref": round(ncc_ll, 4),
                "nrmse_vs_ref": round(nrmse_ll, 4),
            },
            {
                "method": "Nonlinear least-squares T2 fit",
                "ncc_vs_ref": round(ncc_nls, 4),
                "nrmse_vs_ref": round(nrmse_nls, 4),
            },
        ],
        "ncc_boundary": round(0.9 * ncc_nls, 4),
        "nrmse_boundary": round(1.1 * nrmse_nls, 4),
    }
    metrics_path = os.path.join(TASK_DIR, 'evaluation', 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Visualizations
    plot_t2_maps(
        T2_gt_2d, T2_ll, mask_2d,
        title_est="Log-Linear T2",
        save_path=os.path.join(output_dir, 'T2_loglinear.png'),
    )
    plot_t2_maps(
        T2_gt_2d, T2_nls, mask_2d,
        title_est="Nonlinear LS T2",
        save_path=os.path.join(output_dir, 'T2_nonlinear.png'),
    )

    # Signal decay curves at a few representative pixels
    # Find pixels near center with different T2 values
    cy, cx = signal_2d.shape[0] // 2, signal_2d.shape[1] // 2
    pixel_coords = []
    for dy, dx in [(-30, 0), (0, 0), (30, 0), (0, 40)]:
        py, px = cy + dy, cx + dx
        if mask_2d[py, px]:
            pixel_coords.append((py, px))
    if pixel_coords:
        plot_signal_decay(
            signal_2d, TE, T2_gt_2d, T2_nls, pixel_coords,
            save_path=os.path.join(output_dir, 'signal_decay.png'),
        )

    print(f"\n  Figures saved to {output_dir}/")
    print("\nDone!")

    return metrics


if __name__ == '__main__':
    main()
