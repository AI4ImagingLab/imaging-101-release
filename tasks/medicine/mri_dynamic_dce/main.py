"""
Dynamic Contrast-Enhanced (DCE) MRI Reconstruction
====================================================

Reconstructs a time-series of MRI images from undersampled Cartesian
k-space using temporal Total Variation regularization via ADMM.

Pipeline:
    1. Load undersampled dynamic k-space and ground truth
    2. Compute zero-filled baseline reconstruction
    3. Run temporal TV reconstruction via ADMM
    4. Evaluate metrics (NRMSE, NCC, PSNR) per frame and overall
    5. Save outputs and generate visualizations

Usage:
    cd tasks/mri_dynamic_dce
    python main.py
"""

import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')

from src.preprocessing import prepare_data
from src.solvers import zero_filled_recon, temporal_tv_pgd
from src.visualization import (
    compute_frame_metrics,
    print_metrics_table,
    plot_frame_comparison,
    plot_time_activity_curves,
    plot_convergence,
    plot_error_maps,
)

TASK_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Solver parameters (not in meta_data.json to avoid leaking algorithm info) ──
_TV_LAMBDA = 0.001
_PGD_MAX_ITER = 200
_PGD_TOL = 1e-5


def main():
    print('=' * 60)
    print('DCE-MRI: Temporal TV Reconstruction')
    print('=' * 60)

    data_dir = os.path.join(TASK_DIR, 'data')
    output_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load data ──────────────────────────────────────────────
    print('\n[1/5] Loading data ...')
    obs, gt, meta = prepare_data(data_dir)
    kspace = obs['undersampled_kspace']
    masks = obs['undersampling_masks']
    gt_images = gt['dynamic_images']
    time_points = gt['time_points']

    T, N, _ = kspace.shape
    actual_rate = masks.mean()
    print(f'  {T} frames, {N}x{N} resolution')
    print(f'  Actual sampling rate: {actual_rate:.1%}')

    # ── Step 2: Zero-filled baseline ───────────────────────────────────
    print('\n[2/5] Computing zero-filled baseline ...')
    zf_recon = zero_filled_recon(kspace)
    zf_metrics = compute_frame_metrics(zf_recon, gt_images)
    print('  Zero-filled metrics:')
    print(f'    Avg NRMSE: {zf_metrics["avg_nrmse"]:.4f}')
    print(f'    Avg NCC:   {zf_metrics["avg_ncc"]:.4f}')
    print(f'    Avg PSNR:  {zf_metrics["avg_psnr"]:.2f} dB')

    # ── Step 3: Temporal TV reconstruction ─────────────────────────────
    print(f'\n[3/5] Running temporal TV via PGD (lambda={_TV_LAMBDA}, '
          f'max_iter={_PGD_MAX_ITER}) ...')
    tv_recon, tv_info = temporal_tv_pgd(
        kspace, masks,
        lamda=_TV_LAMBDA,
        max_iter=_PGD_MAX_ITER,
        tol=_PGD_TOL,
        verbose=True,
    )

    # ── Step 4: Evaluate ───────────────────────────────────────────────
    print('\n[4/5] Computing metrics ...')
    tv_metrics = compute_frame_metrics(tv_recon, gt_images)
    print('\n  Temporal TV metrics (per frame):')
    print_metrics_table(tv_metrics)
    print(f'\n  Overall NRMSE: {tv_metrics["overall_nrmse"]:.4f}')
    print(f'  Overall NCC:   {tv_metrics["overall_ncc"]:.4f}')

    # ── Step 5: Save outputs and visualizations ────────────────────────
    print('\n[5/5] Saving outputs ...')

    # Save reconstructions
    np.savez_compressed(
        os.path.join(output_dir, 'tv_reconstruction.npz'),
        reconstruction=tv_recon.astype(np.float32),
    )
    np.savez_compressed(
        os.path.join(output_dir, 'zero_filled.npz'),
        reconstruction=zf_recon.astype(np.float32),
    )

    # Save metrics
    metrics_detail = {
        'temporal_tv': {
            'avg_nrmse': tv_metrics['avg_nrmse'],
            'avg_ncc': tv_metrics['avg_ncc'],
            'avg_psnr': tv_metrics['avg_psnr'],
            'overall_nrmse': tv_metrics['overall_nrmse'],
            'overall_ncc': tv_metrics['overall_ncc'],
            'per_frame': tv_metrics['per_frame'],
        },
        'zero_filled': {
            'avg_nrmse': zf_metrics['avg_nrmse'],
            'avg_ncc': zf_metrics['avg_ncc'],
            'avg_psnr': zf_metrics['avg_psnr'],
            'overall_nrmse': zf_metrics['overall_nrmse'],
            'overall_ncc': zf_metrics['overall_ncc'],
        },
        'solver_params': {
            'lambda': _TV_LAMBDA,
            'max_iter': _PGD_MAX_ITER,
            'num_iter': tv_info['num_iter'],
        },
    }
    with open(os.path.join(output_dir, 'metrics_detail.json'), 'w') as f:
        json.dump(metrics_detail, f, indent=2)

    # Save convergence info
    np.savez_compressed(
        os.path.join(output_dir, 'convergence.npz'),
        loss_history=np.array(tv_info['loss_history']),
    )

    # Visualizations
    plot_frame_comparison(
        gt_images, zf_recon, tv_recon,
        save_path=os.path.join(output_dir, 'frame_comparison.png'),
    )
    plot_error_maps(
        gt_images, tv_recon,
        save_path=os.path.join(output_dir, 'error_maps.png'),
    )
    plot_convergence(
        tv_info['loss_history'],
        save_path=os.path.join(output_dir, 'convergence.png'),
    )

    # Time-activity curves for dynamic regions
    # Region 1 center: phantom center is (64,64), region1 at (cx-15, cy+15)=(49, 79)
    plot_time_activity_curves(
        gt_images, zf_recon, tv_recon, time_points,
        roi_center=(49, 79), roi_radius=4,
        save_path=os.path.join(output_dir, 'tac_region1.png'),
    )
    # Region 2 center: (cx+12, cy-12) = (76, 52)
    plot_time_activity_curves(
        gt_images, zf_recon, tv_recon, time_points,
        roi_center=(76, 52), roi_radius=4,
        save_path=os.path.join(output_dir, 'tac_region2.png'),
    )

    print(f'  Saved to {output_dir}/')
    print('\nDone!')


if __name__ == '__main__':
    main()
