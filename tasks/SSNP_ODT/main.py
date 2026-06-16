"""
SSNP Intensity Diffraction Tomography
=======================================

Reconstruction pipeline for 3D RI from multi-angle intensity measurements.

Usage
-----
    cd tasks/SSNP_ODT
    python main.py
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from src.physics_model import SSNPConfig, SSNPForwardModel
from src.solvers import SSNPReconstructor
from src.visualization import (
    plot_ri_slices,
    plot_xz_cross_section,
    plot_comparison,
    plot_loss_history,
    plot_measurements,
    compute_metrics,
    print_metrics_table,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("Step 1: Loading data and metadata...")
    with open("data/meta_data.json") as f:
        metadata = json.load(f)

    raw = np.load("data/raw_data.npz")
    gt  = np.load("data/ground_truth.npz")
    meas_np    = raw["measurements"][0].astype(np.float64)   # (n_angles, Ny, Nx)
    phantom_dn = gt["delta_n"][0].astype(np.float64)          # (Nz, Ny, Nx)

    config = SSNPConfig.from_metadata(metadata)
    print(f"  Measurements   : {meas_np.shape}  range [{meas_np.min():.4f}, {meas_np.max():.4f}]")
    print(f"  Ground truth   : {phantom_dn.shape}  Δn [{phantom_dn.min():.6f}, {phantom_dn.max():.6f}]")
    print(f"  Norm. res      : {config.res}")
    print(f"  Voxel size     : {config.res_um} μm")

    # ── Step 2: Build forward model ───────────────────────────────────────
    print("\nStep 2: Building SSNP forward model...")
    model = SSNPForwardModel(config, device=device)
    print(f"  {model}")

    # ── Step 3: Reconstruct ───────────────────────────────────────────────
    print("\nStep 3: Reconstructing 3D RI...")
    reconstructor = SSNPReconstructor(
        n_iter=100,
        lr=50.0,
        tv_weight=0.0,
        positivity=True,
        device=device,
    )

    measurements = torch.tensor(meas_np, dtype=torch.float64, device=device)
    meas_amp = torch.sqrt(measurements)

    dn_recon, loss_history = reconstructor.reconstruct(meas_amp, model)
    print(f"  Final loss: {loss_history[-1]:.6f}")

    # ── Step 4: Evaluate ──────────────────────────────────────────────────
    print("\nStep 4: Evaluating reconstruction quality...")
    metrics = compute_metrics(dn_recon, phantom_dn)
    print()
    print_metrics_table(metrics)

    with open("evaluation/metrics.json") as f:
        eval_cfg = json.load(f)
    ncc_ok   = metrics["ncc"]   >= eval_cfg["ncc_boundary"]
    nrmse_ok = metrics["nrmse"] <= eval_cfg["nrmse_boundary"]
    print(f"\n  NCC   {metrics['ncc']:.4f} >= {eval_cfg['ncc_boundary']}  : {'PASS' if ncc_ok   else 'FAIL'}")
    print(f"  NRMSE {metrics['nrmse']:.4f} <= {eval_cfg['nrmse_boundary']}  : {'PASS' if nrmse_ok else 'FAIL'}")

    # ── Step 5: Save outputs ──────────────────────────────────────────────
    print("\nStep 5: Saving outputs...")
    ref_dir = os.path.join("evaluation", "reference_outputs")
    os.makedirs(ref_dir, exist_ok=True)

    np.save(os.path.join(ref_dir, "reconstruction.npy"), dn_recon)
    np.save(os.path.join(ref_dir, "loss_history.npy"), np.array(loss_history))

    fig = plot_ri_slices(phantom_dn, title="Ground Truth Δn")
    fig.savefig(os.path.join(ref_dir, "ground_truth_slices.png"), dpi=150)

    fig = plot_ri_slices(dn_recon, title="Reconstructed Δn")
    fig.savefig(os.path.join(ref_dir, "reconstruction_slices.png"), dpi=150)

    fig = plot_measurements(meas_np)
    fig.savefig(os.path.join(ref_dir, "measurements.png"), dpi=150)

    fig = plot_comparison(phantom_dn, dn_recon)
    fig.savefig(os.path.join(ref_dir, "comparison.png"), dpi=150)

    fig = plot_loss_history(loss_history)
    fig.savefig(os.path.join(ref_dir, "loss_history.png"), dpi=150)

    print("\nDone. Outputs saved to evaluation/reference_outputs/")
    return dn_recon, metrics


if __name__ == "__main__":
    main()
