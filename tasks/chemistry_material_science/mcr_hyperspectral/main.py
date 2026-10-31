"""MCR Hyperspectral Reconstruction Pipeline.

Runs five MCR-AR variants on a synthetic hyperspectral image and
evaluates their accuracy against known ground truth.

Usage:
    python main.py
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.preprocessing import load_observation, load_ground_truth, load_metadata, estimate_initial_spectra
from src.solvers import run_all_methods
from src.visualization import (
    compute_metrics,
    compute_method_metrics,
    plot_spectral_components,
    plot_concentration_maps,
    plot_comparison_boxplots,
    plot_method_result,
)

DATA_DIR = pathlib.Path(__file__).parent / "data"
OUTPUT_DIR = pathlib.Path(__file__).parent / "output"
REF_DIR = pathlib.Path(__file__).parent / "evaluation" / "reference_outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("Step 1: Loading data...")
    obs = load_observation(DATA_DIR)
    gt = load_ground_truth(DATA_DIR)
    meta = load_metadata(DATA_DIR)

    hsi_noisy = obs["hsi_noisy"]
    wn = obs["wn"]
    conc_ravel = gt["concentrations_ravel"]
    spectra = gt["spectra"]
    conc = gt["concentrations"]
    n_components = meta["n_components"]
    M, N = meta["M"], meta["N"]

    # ------------------------------------------------------------------
    # Step 2: Estimate initial spectra (SVD)
    # ------------------------------------------------------------------
    print("Step 2: Estimating initial spectra via SVD...")
    initial_spectra = estimate_initial_spectra(hsi_noisy, n_components)

    # ------------------------------------------------------------------
    # Step 3: Run all MCR methods
    # ------------------------------------------------------------------
    print("Step 3: Running MCR methods...")
    results = run_all_methods(hsi_noisy, initial_spectra, conc_ravel, spectra)

    # ------------------------------------------------------------------
    # Step 4: Compute metrics
    # ------------------------------------------------------------------
    print("Step 4: Computing metrics...")
    method_names = [r["name"] for r in results]
    all_metrics = {}
    for r in results:
        m = compute_method_metrics(r, conc_ravel, spectra, hsi_noisy)
        all_metrics[r["name"]] = m
        print("  {}: conc_ncc={:.4f}  conc_nrmse={:.4f}  spec_ncc={:.4f}  mse={:.2e}".format(
            r["name"], m["conc_ncc"], m["conc_nrmse"], m["spec_ncc"], m["mse"]))

    # ------------------------------------------------------------------
    # Step 5: Save outputs
    # ------------------------------------------------------------------
    print("Step 5: Saving outputs...")

    # Save per-method reconstructions
    for r in results:
        sel = r["select"]
        np.savez(
            OUTPUT_DIR / "{}.npz".format(r["name"].replace(" ", "_")),
            C_opt=r["mcr"].C_opt_[:, sel],
            ST_opt=r["mcr"].ST_opt_[sel, :],
            D_opt=r["mcr"].D_opt_,
            err=np.array(r["mcr"].err),
            select=np.array(sel),
        )

    # Save metrics summary
    with open(OUTPUT_DIR / "metrics_summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ------------------------------------------------------------------
    # Step 6: Generate figures
    # ------------------------------------------------------------------
    print("Step 6: Generating figures...")

    # True spectral components
    fig, ax = plt.subplots(figsize=(4, 2.25))
    plot_spectral_components(ax, wn, spectra, title="True Spectral Components")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "true_spectra.png", dpi=300)
    plt.close(fig)

    # True concentration maps
    fig = plt.figure(figsize=(8, 2.5))
    plot_concentration_maps(fig, conc, suptitle="True Concentrations")
    fig.savefig(OUTPUT_DIR / "true_concentrations.png", dpi=300)
    plt.close(fig)

    # Initial spectra guess
    fig, ax = plt.subplots(figsize=(4, 2.25))
    plot_spectral_components(ax, wn, initial_spectra, title="Initial Spectra (SVD)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "initial_spectra.png", dpi=300)
    plt.close(fig)

    # Comparison boxplots
    fig = plot_comparison_boxplots(results, conc_ravel, spectra, hsi_noisy, method_names)
    fig.savefig(OUTPUT_DIR / "comparison_panel.png", dpi=300)
    plt.close(fig)

    # Per-method results
    for r in results:
        fig = plot_method_result(wn, r, (M, N), r["name"])
        fig.savefig(OUTPUT_DIR / "{}_result.png".format(r["name"].replace(" ", "_")), dpi=300)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Step 7: Save reference outputs (for evaluation harness)
    # ------------------------------------------------------------------
    REF_DIR.mkdir(parents=True, exist_ok=True)

    # Use MCR-NNLS as primary reference (best MSE in original paper)
    ref_result = next(r for r in results if r["name"] == "MCR-NNLS")
    sel = ref_result["select"]
    np.savez(
        REF_DIR / "reconstruction.npz",
        C_opt=ref_result["mcr"].C_opt_[:, sel],
        ST_opt=ref_result["mcr"].ST_opt_[sel, :],
        D_opt=ref_result["mcr"].D_opt_,
    )

    # Save all methods' metrics for reference
    with open(REF_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\nDone. Outputs saved to: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
