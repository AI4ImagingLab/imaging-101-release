"""
CG-SENSE Multi-Coil MRI Reconstruction
========================================

Pipeline:
    1. Load multi-coil k-space, sensitivity maps, phantom
    2. Undersample k-space (R=4, 16-line ACS)
    3. Reconstruct via CG-SENSE (conjugate gradient on normal equations)
    4. Compare with zero-filled reconstruction
    5. Visualize results and save outputs
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.preprocessing import prepare_data
from src.physics_model import zero_filled_recon
from src.solvers import cgsense_image_recon
from src.visualization import (
    compute_metrics, print_metrics,
    plot_reconstruction_comparison, plot_error_maps,
    plot_sensitivity_maps,
)

# ── Solver parameters (not in meta_data.json) ──
_R = 4
_ACS_WIDTH = 16


def main(data_dir: str = "data", output_dir: str = "evaluation/reference_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load data ──
    print("Step 1: Loading data...")
    kspace_us, sens, kspace_full, phantom, metadata = prepare_data(
        data_dir, R=_R, acs_width=_ACS_WIDTH,
    )
    print(f"  K-space: {kspace_full.shape} ({metadata['n_coils']} coils)")
    print(f"  Acceleration: R={_R}, ACS={_ACS_WIDTH} lines")

    # Ground truth (normalized phantom)
    gt = phantom / phantom.max() if phantom.max() > 0 else phantom

    # ── Step 2: Zero-filled baseline ──
    print("\nStep 2: Zero-filled reconstruction...")
    zf = zero_filled_recon(kspace_us)
    zf_norm = zf / zf.max() if zf.max() > 0 else zf

    # ── Step 3: CG-SENSE reconstruction ──
    print(f"\nStep 3: CG-SENSE reconstruction...")
    recon = cgsense_image_recon(kspace_us, sens)
    print(f"  Reconstruction: {recon.shape}")

    # Save outputs
    np.savez_compressed(
        os.path.join(output_dir, "sense_reconstruction.npz"),
        reconstruction=recon[None, ...].astype(np.float32),
    )
    np.savez_compressed(
        os.path.join(output_dir, "zerofill.npz"),
        reconstruction=zf_norm[None, ...].astype(np.float32),
    )
    np.save(os.path.join(output_dir, "fully_sampled.npy"), gt.astype(np.float32))

    # ── Step 4: Evaluate ──
    print("\nStep 4: Computing metrics...")
    m_sense = compute_metrics(recon, gt)
    m_zf = compute_metrics(zf_norm, gt)
    print_metrics(m_sense, "CG-SENSE")
    print_metrics(m_zf, "Zero-fill")

    metrics_out = {"sense": m_sense, "zerofill": m_zf}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ── Step 5: Visualize ──
    print("\nStep 5: Generating visualizations...")
    plot_reconstruction_comparison(
        recon, zf_norm, gt,
        save_path=os.path.join(output_dir, "reconstruction_comparison.png"),
    )
    plot_error_maps(
        recon, zf_norm, gt,
        save_path=os.path.join(output_dir, "error_maps.png"),
    )
    plot_sensitivity_maps(
        sens,
        save_path=os.path.join(output_dir, "sensitivity_maps.png"),
    )
    print(f"  Saved figures to {output_dir}")

    print("\nDone!")
    return recon, metrics_out


if __name__ == "__main__":
    main()
