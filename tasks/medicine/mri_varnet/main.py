"""
End-to-End VarNet Multi-Coil MRI Reconstruction
=================================================

Pipeline:
    1. Load multi-coil k-space and RSS ground truth
    2. Load pretrained VarNet (12 cascades)
    3. Apply 4x equispaced undersampling + VarNet inference
    4. Evaluate against RSS ground truth
    5. Visualize results and save outputs
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.preprocessing import prepare_data
from src.solvers import load_varnet, varnet_reconstruct_batch
from src.visualization import (
    compute_metrics, print_metrics,
    plot_reconstruction_comparison, plot_error_maps,
)

# ── Solver parameters (not in meta_data.json) ──
_ACCELERATION = 4
_CENTER_FRACTION = 0.08
_DEVICE = "cpu"


def main(data_dir: str = "data", output_dir: str = "evaluation/reference_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load data ──
    print("Step 1: Loading data...")
    kspace, gt, metadata = prepare_data(data_dir)
    print(f"  K-space: {kspace.shape} ({metadata['n_coils']} coils)")
    print(f"  Ground truth: {gt.shape}")

    # ── Step 2: Load model ──
    print("\nStep 2: Loading pretrained VarNet...")
    weights_path = os.path.join(data_dir, "varnet_knee_state_dict.pt")
    model = load_varnet(weights_path, device=_DEVICE)
    print("  VarNet loaded (12 cascades, 18 channels)")

    # ── Step 3: Reconstruct ──
    target_h, target_w = metadata["image_shape"]
    print(f"\nStep 3: VarNet reconstruction ({_ACCELERATION}x acceleration)...")
    recons, zerofills = varnet_reconstruct_batch(
        model, kspace, _ACCELERATION, _CENTER_FRACTION,
        target_h, target_w, _DEVICE,
    )
    print(f"  Reconstructions: {recons.shape}")

    # Save outputs
    np.savez_compressed(os.path.join(output_dir, "varnet_reconstruction.npz"),
                        reconstruction=recons.astype(np.float32))
    np.savez_compressed(os.path.join(output_dir, "zerofill.npz"),
                        reconstruction=zerofills.astype(np.float32))

    # ── Step 4: Evaluate ──
    print("\nStep 4: Computing metrics...")
    all_m_vn, all_m_zf = [], []
    for i in range(recons.shape[0]):
        m_vn = compute_metrics(recons[i], gt[i])
        m_zf = compute_metrics(zerofills[i], gt[i])
        all_m_vn.append(m_vn)
        all_m_zf.append(m_zf)
        print(f"  Slice {i}: VarNet SSIM={m_vn['ssim']:.4f} | ZF SSIM={m_zf['ssim']:.4f}")

    avg_vn = {k: float(np.mean([m[k] for m in all_m_vn])) for k in all_m_vn[0]}
    avg_zf = {k: float(np.mean([m[k] for m in all_m_zf])) for k in all_m_zf[0]}
    print(f"\n  Average:")
    print_metrics(avg_vn, "VarNet")
    print_metrics(avg_zf, "Zero-fill")

    metrics_out = {"varnet_avg": avg_vn, "zerofill_avg": avg_zf}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ── Step 5: Visualize ──
    print("\nStep 5: Generating visualizations...")
    plot_reconstruction_comparison(
        recons, zerofills, gt, slice_idx=0,
        save_path=os.path.join(output_dir, "reconstruction_comparison.png"),
    )
    plot_error_maps(
        recons[0], zerofills[0], gt[0],
        save_path=os.path.join(output_dir, "error_maps.png"),
    )
    print(f"  Saved figures to {output_dir}")

    print("\nDone!")
    return recons, metrics_out


if __name__ == "__main__":
    main()
