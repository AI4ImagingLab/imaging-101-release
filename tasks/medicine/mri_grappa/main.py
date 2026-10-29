"""
GRAPPA Multi-Coil MRI Reconstruction
======================================

Pipeline:
    1. Load synthetic multi-coil k-space and phantom
    2. Undersample k-space (R=2, 20-line ACS)
    3. Reconstruct via GRAPPA kernel calibration + interpolation
    4. Compare with fully-sampled reference and zero-fill
    5. Visualize results and save outputs
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.preprocessing import prepare_data
from src.physics_model import fully_sampled_recon, zero_filled_recon
from src.solvers import grappa_reconstruct, grappa_image_recon
from src.visualization import (
    compute_metrics, print_metrics,
    plot_reconstruction_comparison, plot_error_maps,
    plot_kspace,
)

# ── Solver parameters (not in meta_data.json) ──
_R = 2
_ACS_WIDTH = 20
_KERNEL_SIZE = (5, 5)
_LAMDA = 0.01


def main(data_dir: str = "data", output_dir: str = "evaluation/reference_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load data ──
    print("Step 1: Loading data...")
    kspace_us, calib, kspace_full, phantom, metadata = prepare_data(
        data_dir, R=_R, acs_width=_ACS_WIDTH,
    )
    print(f"  K-space: {kspace_full.shape}, Calib: {calib.shape}")
    print(f"  Acceleration: R={_R}, ACS={_ACS_WIDTH} lines")

    # ── Step 2: Reference and zero-fill ──
    print("\nStep 2: Computing reference and zero-fill...")
    ref_img = fully_sampled_recon(kspace_full)
    zf_img = zero_filled_recon(kspace_us)

    # ── Step 3: GRAPPA reconstruction ──
    print(f"\nStep 3: GRAPPA reconstruction (kernel={_KERNEL_SIZE}, lambda={_LAMDA})...")
    kspace_grappa = grappa_reconstruct(kspace_us, calib, _KERNEL_SIZE, _LAMDA)
    recon_img = grappa_image_recon(kspace_us, calib, _KERNEL_SIZE, _LAMDA)
    print(f"  Reconstruction range: [{recon_img.min():.4f}, {recon_img.max():.4f}]")

    # Save outputs
    np.savez_compressed(
        os.path.join(output_dir, "grappa_reconstruction.npz"),
        reconstruction=recon_img[None, ...].astype(np.float32),
    )
    np.savez_compressed(
        os.path.join(output_dir, "zerofill.npz"),
        reconstruction=zf_img[None, ...].astype(np.float32),
    )
    np.save(os.path.join(output_dir, "ground_truth.npy"), ref_img.astype(np.float32))

    # ── Step 4: Evaluate ──
    print("\nStep 4: Computing metrics...")
    m_grappa = compute_metrics(recon_img, ref_img)
    m_zf = compute_metrics(zf_img, ref_img)
    print_metrics(m_grappa, "GRAPPA")
    print_metrics(m_zf, "Zero-fill")

    metrics_out = {"grappa": m_grappa, "zerofill": m_zf}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ── Step 5: Visualize ──
    print("\nStep 5: Generating visualizations...")
    plot_reconstruction_comparison(
        recon_img, zf_img, ref_img,
        save_path=os.path.join(output_dir, "reconstruction_comparison.png"),
    )
    plot_error_maps(
        recon_img, zf_img, ref_img,
        save_path=os.path.join(output_dir, "error_maps.png"),
    )
    plot_kspace(
        kspace_us, kspace_grappa,
        save_path=os.path.join(output_dir, "kspace_comparison.png"),
    )
    print(f"  Saved figures to {output_dir}")

    print("\nDone!")
    return recon_img, metrics_out


if __name__ == "__main__":
    main()
