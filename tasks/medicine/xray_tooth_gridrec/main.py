"""X-ray CT reconstruction of a tooth cross-section using Filtered Back-Projection.

Pipeline:
1. Load raw projection data (detector counts, flat/dark fields, angles)
2. Preprocess: flat-field correction, -log linearization
3. Find rotation center
4. Reconstruct via Filtered Back-Projection (FBP)
5. Apply circular mask
6. Compute metrics against baseline reference
7. Save outputs
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add task root to path for src imports
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import load_observation, load_metadata, normalize, minus_log
from src.physics_model import find_rotation_center
from src.solvers import filtered_back_projection, circular_mask
from src.visualization import compute_metrics, plot_sinogram, plot_reconstruction

# ── Pipeline parameters (not in meta_data.json per spec) ──────────────
_ROT_CENTER_INIT = 290
_ROT_CENTER_TOL = 0.5
_CIRC_MASK_RATIO = 0.95


def main():
    data_dir = os.path.join(TASK_DIR, "data")
    output_dir = os.path.join(TASK_DIR, "output")
    ref_dir = os.path.join(TASK_DIR, "evaluation", "reference_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    print("Loading data...")
    obs = load_observation(data_dir)
    meta = load_metadata(data_dir)
    proj = obs["projections"]
    flat = obs["flat_field"]
    dark = obs["dark_field"]
    theta = obs["theta"]
    print(f"  Projections: {proj.shape}, Angles: {len(theta)}")

    # 2. Preprocess
    print("Preprocessing...")
    proj_norm = normalize(proj, flat, dark)
    sinogram_data = minus_log(proj_norm)
    print(f"  Sinogram shape: {sinogram_data.shape}")

    # 3. Find rotation center (using first sinogram row)
    print("Finding rotation center...")
    sino_slice = sinogram_data[:, 0, :]
    n_det = meta["n_detector_pixels"]
    rot_center = find_rotation_center(
        sino_slice, theta, init=_ROT_CENTER_INIT, tol=_ROT_CENTER_TOL,
    )
    print(f"  Rotation center: {rot_center:.2f}")

    # 4. Reconstruct each sinogram slice
    print("Reconstructing...")
    n_slices = sinogram_data.shape[1]
    recon_slices = []
    for s in range(n_slices):
        sino = sinogram_data[:, s, :]
        # Shift sinogram to align rotation center with detector midpoint
        from src.physics_model import _shift_sinogram
        sino_shifted = _shift_sinogram(sino, rot_center, n_det)
        recon = filtered_back_projection(sino_shifted, theta, n_det)
        recon_slices.append(recon)
    recon_volume = np.stack(recon_slices, axis=0)
    print(f"  Reconstruction shape: {recon_volume.shape}")

    # 5. Apply circular mask
    for s in range(n_slices):
        recon_volume[s] = circular_mask(recon_volume[s], ratio=_CIRC_MASK_RATIO)

    # 6. Save reconstruction
    np.save(os.path.join(output_dir, "reconstruction.npy"), recon_volume)
    print(f"  Saved reconstruction to output/reconstruction.npy")

    # 7. Compute metrics against reference if available
    ref_path = os.path.join(ref_dir, "reconstruction.npy")
    if os.path.exists(ref_path):
        ref_recon = np.load(ref_path)
        metrics = compute_metrics(recon_volume[0], ref_recon[0])
        print(f"  Metrics vs reference: NRMSE={metrics['nrmse']:.4f}, NCC={metrics['ncc']:.4f}")

        metrics_out = {
            "fbp_vs_ref": {
                "nrmse_vs_ref": metrics["nrmse"],
                "ncc_vs_ref": metrics["ncc"],
            }
        }
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics_out, f, indent=2)
    else:
        print("  No reference outputs found, skipping metrics.")

    # 8. Save visualizations
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_sinogram(sinogram_data[:, 0, :], ax=axes[0], title="Sinogram (slice 0)")
    plot_reconstruction(recon_volume[0], ax=axes[1], title="FBP Reconstruction (slice 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "results.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved results.png")

    print("Done.")


if __name__ == "__main__":
    main()
