"""
PnP-ADMM Compressed Sensing MRI Reconstruction
=================================================

Pipeline:
    1. Load brain MRI image, undersampling mask, noise
    2. Load pretrained RealSN-DnCNN denoiser
    3. Reconstruct via PnP-ADMM (100 iterations)
    4. Evaluate against ground truth
    5. Visualize results and save outputs
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.preprocessing import prepare_data
from src.denoiser import load_denoiser
from src.solvers import pnp_admm_reconstruct
from src.visualization import (
    compute_metrics, print_metrics,
    plot_reconstruction_comparison, plot_error_maps,
    plot_psnr_convergence, plot_mask,
)

# ── Solver parameters (not in meta_data.json) ──
_ALPHA = 2.0
_SIGMA = 15
_MAXITR = 100
_DEVICE = "cpu"


def main(data_dir: str = "data", output_dir: str = "evaluation/reference_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load data ──
    print("Step 1: Loading data...")
    im_orig, mask, noises, metadata = prepare_data(data_dir, mask_name="random")
    print(f"  Image: {im_orig.shape}, range: [{im_orig.min():.3f}, {im_orig.max():.3f}]")
    print(f"  Mask: {mask.sum()/mask.size*100:.1f}% sampling")

    # ── Step 2: Load denoiser ──
    print("\nStep 2: Loading RealSN-DnCNN denoiser (sigma=15)...")
    weights_path = os.path.join(data_dir, "RealSN_DnCNN_noise15.pth")
    model = load_denoiser(weights_path, device=_DEVICE)
    print("  Denoiser loaded")

    # ── Step 3: Reconstruct ──
    print(f"\nStep 3: PnP-ADMM reconstruction "
          f"(alpha={_ALPHA}, sigma={_SIGMA}, {_MAXITR} iterations)...")
    result = pnp_admm_reconstruct(
        model, im_orig, mask, noises,
        alpha=_ALPHA, sigma=_SIGMA, maxitr=_MAXITR, device=_DEVICE,
    )
    recon = result["reconstruction"]
    zerofill = result["zerofill"]
    print(f"  Reconstruction range: [{recon.min():.4f}, {recon.max():.4f}]")
    print(f"  Final PSNR: {result['psnr_history'][-1]:.2f} dB")

    # Save outputs
    np.savez_compressed(
        os.path.join(output_dir, "pnp_admm_reconstruction.npz"),
        reconstruction=recon[None, ...].astype(np.float32),
    )
    np.savez_compressed(
        os.path.join(output_dir, "zerofill.npz"),
        zerofill=zerofill[None, ...].astype(np.float32),
    )
    np.save(os.path.join(output_dir, "psnr_history.npy"), result["psnr_history"])
    np.save(os.path.join(output_dir, "ground_truth.npy"), im_orig.astype(np.float32))

    # ── Step 4: Evaluate ──
    print("\nStep 4: Computing metrics...")
    m_pnp = compute_metrics(recon, im_orig)
    m_zf = compute_metrics(zerofill, im_orig)
    print_metrics(m_pnp, "PnP-ADMM")
    print_metrics(m_zf, "Zero-fill")

    metrics_out = {
        "pnp_admm": m_pnp,
        "zerofill": m_zf,
        "alpha": _ALPHA,
        "sigma": _SIGMA,
        "maxitr": _MAXITR,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ── Step 5: Visualize ──
    print("\nStep 5: Generating visualizations...")
    plot_reconstruction_comparison(
        recon, zerofill, im_orig,
        save_path=os.path.join(output_dir, "reconstruction_comparison.png"),
    )
    plot_error_maps(
        recon, zerofill, im_orig,
        save_path=os.path.join(output_dir, "error_maps.png"),
    )
    plot_psnr_convergence(
        result["psnr_history"],
        save_path=os.path.join(output_dir, "psnr_convergence.png"),
    )
    plot_mask(
        mask, title="Random 30% Mask",
        save_path=os.path.join(output_dir, "mask.png"),
    )
    print(f"  Saved figures to {output_dir}")

    print("\nDone!")
    return recon, metrics_out


if __name__ == "__main__":
    main()
