"""
Photoacoustic tomography reconstruction pipeline.

Generates synthetic PA data from spherical absorbers, reconstructs the initial
pressure distribution using the Xu & Wang (2005) universal back-projection
algorithm, and evaluates reconstruction quality.
"""

import matplotlib
matplotlib.use('Agg')

import json
import os
import time
import numpy as np

from src.generate_data import generate_and_save
from src.preprocessing import load_raw_data, load_ground_truth, load_metadata
from src.solvers import universal_back_projection
from src.visualization import (
    compute_ncc, compute_nrmse, centre_crop,
    plot_reconstruction, plot_cross_sections, plot_signals,
)

# ---- Named constants for solver/algorithm parameters ----
_RECONSTRUCTION_RESOLUTION_M = 500e-6
_NFFT = 2048
_DYNAMIC_RANGE_DB = 6


def main():
    data_dir = "data"
    ref_dir = "evaluation/reference_outputs"
    os.makedirs(ref_dir, exist_ok=True)

    # Step 1: Generate synthetic data (if not already present)
    if not os.path.exists(f"{data_dir}/raw_data.npz"):
        print("=== Generating synthetic PA data ===")
        generate_and_save(data_dir)
    else:
        print("=== Data already exists, skipping generation ===")

    # Step 2: Load data
    print("=== Loading data ===")
    signals, xd, yd, t = load_raw_data(data_dir)
    gt_image, image_x, image_y = load_ground_truth(data_dir)
    meta = load_metadata(data_dir)

    c = meta["sound_speed_m_per_s"]
    z_target = meta["target_plane_z_m"]

    print(f"Signals shape: {signals.shape}")
    print(f"Detector array: {len(xd)}x{len(yd)}")
    print(f"Time samples: {len(t)}")

    # Step 3: Reconstruct via universal back-projection
    print("=== Reconstructing via universal back-projection ===")
    t_start = time.time()
    recon, xf, yf, zf = universal_back_projection(
        signals, xd, yd, t, z_target, c=c,
        resolution=_RECONSTRUCTION_RESOLUTION_M,
        det_area=(meta["detector_size_m"])**2,
        nfft=_NFFT,
    )
    elapsed = time.time() - t_start
    print(f"Reconstruction completed in {elapsed:.1f} s")

    recon_2d = np.squeeze(recon)
    print(f"Reconstruction shape: {recon_2d.shape}")

    # Step 4: Evaluate metrics
    print("=== Evaluating reconstruction quality ===")
    recon_crop = centre_crop(recon_2d, fraction=0.8)
    gt_crop = centre_crop(gt_image, fraction=0.8)

    ncc = compute_ncc(recon_crop, gt_crop)
    nrmse = compute_nrmse(recon_crop, gt_crop)
    print(f"NCC:   {ncc:.4f}")
    print(f"NRMSE: {nrmse:.4f}")

    # Step 5: Save reference outputs
    print("=== Saving reference outputs ===")
    np.savez(
        f"{ref_dir}/reconstruction.npz",
        reconstruction=recon_2d[np.newaxis],  # (1, nx, ny)
        image_x=xf[np.newaxis],
        image_y=yf[np.newaxis],
    )
    np.savez(
        f"{ref_dir}/signals.npz",
        signals=signals[np.newaxis],
    )

    # Step 6: Save metrics
    metrics = {
        "baseline": [
            {
                "method": "Universal back-projection (Xu & Wang 2005), 31x31 detectors",
                "ncc_vs_ref": round(ncc, 4),
                "nrmse_vs_ref": round(nrmse, 4),
            }
        ],
        "ncc_boundary": round(0.9 * ncc, 4),
        "nrmse_boundary": round(1.1 * nrmse, 4),
    }
    with open("evaluation/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to evaluation/metrics.json")

    # Step 7: Generate plots
    print("=== Generating plots ===")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plot_reconstruction(recon_2d, xf, yf, ax=ax,
                        dynamic_range_db=_DYNAMIC_RANGE_DB)
    fig.tight_layout()
    fig.savefig(f"{ref_dir}/reconstruction.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_cross_sections(recon_2d, xf, yf, axes=axes)
    fig.tight_layout()
    fig.savefig(f"{ref_dir}/cross_sections.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    mid = len(xd) // 2
    plot_signals(signals, t, xd, yd,
                 det_indices=[(mid, mid), (0, mid), (mid, 0)], ax=ax)
    fig.tight_layout()
    fig.savefig(f"{ref_dir}/signals.png", dpi=150)
    plt.close(fig)

    # Ground truth plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    extent = [image_x[0]*1e3, image_x[-1]*1e3,
              image_y[-1]*1e3, image_y[0]*1e3]
    ax.imshow(gt_image.T, extent=extent, cmap='gray')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Ground Truth')
    fig.tight_layout()
    fig.savefig(f"{ref_dir}/ground_truth.png", dpi=150)
    plt.close(fig)

    print("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
