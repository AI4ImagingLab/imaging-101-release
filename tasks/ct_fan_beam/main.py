"""
Fan-Beam CT Reconstruction
===========================

Reconstructs a 2D image from fan-beam CT sinograms using both
filtered back-projection (FBP) and TV-regularized iterative methods.

Includes both full-scan (360 deg) and short-scan (pi + fan_angle)
geometries, with Parker weighting for the short-scan case.

Pipeline:
1. Load fan-beam sinogram data and metadata
2. FBP reconstruction (full-scan and short-scan with Parker weights)
3. TV-PDHG iterative reconstruction (short-scan)
4. Evaluate against ground truth
5. Save results and visualizations

Usage:
    cd tasks/ct_fan_beam
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
from src.physics_model import fan_beam_geometry, fan_beam_fbp
from src.solvers import solve_tv_pdhg
from src.visualization import (
    compute_ncc,
    compute_nrmse,
    centre_crop_normalize,
    plot_reconstructions,
    plot_sinogram,
)

# Solver parameters
_TV_PDHG_PARAMS = {
    'lam': 0.005,
    'n_iter': 150,
    'positivity': True,
}


def main():
    print("=" * 60)
    print("Fan-Beam CT Reconstruction")
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
    print("\n[1/5] Loading fan-beam CT data...")
    sino_full, sino_short, angles_full, angles_short, det_pos = \
        load_sinogram_data(TASK_DIR)
    phantom = load_ground_truth(TASK_DIR)
    meta = load_metadata(TASK_DIR)

    N = meta['image_size']
    D_sd = meta['source_to_isocenter_pixels']
    D_dd = meta['isocenter_to_detector_pixels']
    n_det = meta['n_det']

    print(f"  Image size: {N}x{N}")
    print(f"  Detector elements: {n_det}")
    print(f"  Full-scan angles: {meta['n_angles_full']} ({360:.0f} deg)")
    print(f"  Short-scan angles: {meta['n_angles_short']} ({meta['short_scan_range_deg']:.1f} deg)")
    print(f"  Fan half-angle: {meta['fan_half_angle_deg']:.1f} deg")
    print(f"  Source-isocenter: {D_sd}, Isocenter-detector: {D_dd}")

    # Remove batch dims
    sino_full_2d = preprocess_sinogram(sino_full)
    sino_short_2d = preprocess_sinogram(sino_short)
    phantom_2d = phantom[0]

    # Build geometry objects
    geo_full = fan_beam_geometry(N, n_det, len(angles_full), D_sd, D_dd,
                                 angle_range=2 * np.pi)
    short_range = meta['short_scan_range_deg'] * np.pi / 180
    geo_short = fan_beam_geometry(N, n_det, len(angles_short), D_sd, D_dd,
                                  angle_range=short_range)

    # --- Step 2: FBP reconstruction ---
    print("\n[2/5] FBP reconstruction...")

    print("  Full-scan FBP...")
    recon_fbp_full = fan_beam_fbp(sino_full_2d, geo_full, filter_type='hann',
                                   cutoff=0.3, short_scan=False)
    recon_fbp_full = np.maximum(recon_fbp_full, 0)

    print("  Short-scan FBP (with Parker weighting)...")
    recon_fbp_short = fan_beam_fbp(sino_short_2d, geo_short, filter_type='hann',
                                    cutoff=0.3, short_scan=True)
    recon_fbp_short = np.maximum(recon_fbp_short, 0)

    # --- Step 3: TV-PDHG reconstruction (short-scan) ---
    print("\n[3/5] TV-PDHG reconstruction (short-scan)...")
    recon_tv, loss_history = solve_tv_pdhg(
        sino_short_2d, geo_short,
        lam=_TV_PDHG_PARAMS['lam'],
        n_iter=_TV_PDHG_PARAMS['n_iter'],
        positivity=_TV_PDHG_PARAMS['positivity'],
    )

    # --- Step 4: Evaluate ---
    print("\n[4/5] Evaluating results...")

    # Centre-crop and normalize for fair comparison
    gt_crop = centre_crop_normalize(phantom_2d)
    fbp_full_crop = centre_crop_normalize(recon_fbp_full)
    fbp_short_crop = centre_crop_normalize(recon_fbp_short)
    tv_crop = centre_crop_normalize(recon_tv)

    ncc_fbp_full = compute_ncc(fbp_full_crop, gt_crop)
    nrmse_fbp_full = compute_nrmse(fbp_full_crop, gt_crop)
    print(f"  FBP full-scan:  NCC={ncc_fbp_full:.4f}, NRMSE={nrmse_fbp_full:.4f}")

    ncc_fbp_short = compute_ncc(fbp_short_crop, gt_crop)
    nrmse_fbp_short = compute_nrmse(fbp_short_crop, gt_crop)
    print(f"  FBP short-scan: NCC={ncc_fbp_short:.4f}, NRMSE={nrmse_fbp_short:.4f}")

    ncc_tv = compute_ncc(tv_crop, gt_crop)
    nrmse_tv = compute_nrmse(tv_crop, gt_crop)
    print(f"  TV-PDHG short:  NCC={ncc_tv:.4f}, NRMSE={nrmse_tv:.4f}")

    # --- Step 5: Save results ---
    print("\n[5/5] Saving results...")

    output_dir = os.path.join(TASK_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)
    ref_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
    os.makedirs(ref_dir, exist_ok=True)

    # Reference outputs
    np.savez_compressed(
        os.path.join(ref_dir, 'recon_fbp_full.npz'),
        reconstruction=recon_fbp_full[np.newaxis, ...].astype(np.float32),
    )
    np.savez_compressed(
        os.path.join(ref_dir, 'recon_fbp_short.npz'),
        reconstruction=recon_fbp_short[np.newaxis, ...].astype(np.float32),
    )
    np.savez_compressed(
        os.path.join(ref_dir, 'recon_tv_short.npz'),
        reconstruction=recon_tv[np.newaxis, ...].astype(np.float32),
        loss_history=np.array(loss_history, dtype=np.float32),
    )

    # Use best method (TV short-scan) as boundary reference
    best_ncc = max(ncc_tv, ncc_fbp_short)
    best_nrmse = min(nrmse_tv, nrmse_fbp_short)

    metrics = {
        "baseline": [
            {
                "method": "FBP full-scan (360 deg)",
                "ncc_vs_ref": round(ncc_fbp_full, 4),
                "nrmse_vs_ref": round(nrmse_fbp_full, 4),
            },
            {
                "method": "FBP short-scan with Parker weighting",
                "ncc_vs_ref": round(ncc_fbp_short, 4),
                "nrmse_vs_ref": round(nrmse_fbp_short, 4),
            },
            {
                "method": "TV-PDHG short-scan",
                "ncc_vs_ref": round(ncc_tv, 4),
                "nrmse_vs_ref": round(nrmse_tv, 4),
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
    plot_sinogram(sino_full_2d, title="Full-Scan Fan-Beam Sinogram",
                  save_path=os.path.join(output_dir, 'sinogram_full.png'))
    plot_sinogram(sino_short_2d, title="Short-Scan Fan-Beam Sinogram",
                  save_path=os.path.join(output_dir, 'sinogram_short.png'))
    plot_reconstructions(phantom_2d, recon_fbp_full, recon_fbp_full,
                         title_suffix="(Full-Scan)",
                         save_path=os.path.join(output_dir, 'recon_full.png'))
    plot_reconstructions(phantom_2d, recon_fbp_short, recon_tv,
                         title_suffix="(Short-Scan)",
                         save_path=os.path.join(output_dir, 'recon_short.png'))

    print(f"  Figures saved to {output_dir}/")
    print("\nDone!")

    return metrics


if __name__ == '__main__':
    main()
