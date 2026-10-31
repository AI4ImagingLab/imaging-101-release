#!/usr/bin/env python3
"""
Lensless imaging reconstruction pipeline.

Runs ADMM to recover an image from a DiffuserCam measurement.

Usage
-----
    cd tasks/lensless_imaging
    python main.py

Outputs
-------
    evaluation/reference_outputs/  reconstruction.npy, overview.png, reconstruction_display.png
    evaluation/reference_outputs/metrics.json  quality metrics
"""

import json
import os

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import load_npz
from src.solvers import ADMM
from src.visualization import plot_overview, normalise_for_display, gamma_correction

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATA_NPZ    = "data/raw_data.npz"
OUT_DIR     = "evaluation/reference_outputs"
METRICS_PATH = "evaluation/metrics.json"
N_ITER      = 300
MU1         = 1e-6
MU2         = 1e-5
MU3         = 4e-5
TAU         = 1e-4


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data ...")
    psf, data = load_npz(DATA_NPZ)
    print(f"  PSF shape:         {psf.shape}")
    print(f"  Measurement shape: {data.shape}")

    # ------------------------------------------------------------------
    # Set up and run ADMM
    # ------------------------------------------------------------------
    print(f"\nRunning ADMM ({N_ITER} iterations) ...")
    solver = ADMM(psf, mu1=MU1, mu2=MU2, mu3=MU3, tau=TAU)
    solver.set_data(data)
    reconstruction = solver.apply(n_iter=N_ITER, verbose=True)

    print(f"  Reconstruction shape: {reconstruction.shape}")
    print(f"  Value range: [{reconstruction.min():.4f}, {reconstruction.max():.4f}]")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    np.save(os.path.join(OUT_DIR, "reconstruction.npy"), reconstruction)
    print(f"\nSaved reconstruction.npy to {OUT_DIR}/")

    # Overview figure
    fig = plot_overview(psf, data, reconstruction, gamma=2.2)
    fig.savefig(os.path.join(OUT_DIR, "overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Reconstruction only (gamma-corrected)
    rec_disp = gamma_correction(normalise_for_display(reconstruction), gamma=2.2)
    import imageio
    imageio.imwrite(
        os.path.join(OUT_DIR, "reconstruction_display.png"),
        (rec_disp * 255).astype(np.uint8),
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    from src.physics_model import RealFFTConvolve2D
    conv = RealFFTConvolve2D(psf)

    # Reprojection NRMSE: re-apply forward model to reconstruction, compare to measurement.
    # Uses range normalisation (max - min of measurement) per the project standard.
    reprojected = conv.forward(reconstruction)
    reprojection_nrmse = float(
        np.sqrt(np.mean((reprojected - data)**2)) / (data.max() - data.min() + 1e-12)
    )

    print(f"\n  Reprojection NRMSE:    {reprojection_nrmse:.4f}")

    # NCC / NRMSE vs baseline reference reconstruction
    ref = np.load("data/baseline_reference.npz")['reconstruction'][0].astype(np.float64)
    rec64 = reconstruction.astype(np.float64)
    ref_flat, rec_flat = ref.ravel(), rec64.ravel()
    ref_c, rec_c = ref_flat - ref_flat.mean(), rec_flat - rec_flat.mean()
    ncc_vs_ref = float(np.dot(rec_c, ref_c) /
                       (np.linalg.norm(rec_c) * np.linalg.norm(ref_c) + 1e-12))
    nrmse_vs_ref = float(np.sqrt(np.mean((rec_flat - ref_flat)**2)) /
                         (ref_flat.max() - ref_flat.min() + 1e-12))
    print(f"  NCC vs reference:      {ncc_vs_ref:.4f}")
    print(f"  NRMSE vs reference:    {nrmse_vs_ref:.4f}")

    metrics = {
        "baseline": [{
            "method": f"ADMM ({N_ITER} iters, mu1={MU1}, mu2={MU2}, mu3={MU3}, tau={TAU})",
            "ncc_vs_ref": round(ncc_vs_ref, 4),
            "nrmse_vs_ref": round(nrmse_vs_ref, 4),
            "reprojection_nrmse": round(reprojection_nrmse, 4),
        }],
        "ncc_boundary": 0.9,
        "nrmse_boundary": 0.1,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  metrics.json saved to {METRICS_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
