"""Pipeline entry point for InSAR phase unwrapping task.

Runs the full pipeline: load data -> preprocess -> unwrap -> compare -> save.
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np

from src.preprocessing import load_data, load_metadata, extract_phase_and_coherence
from src.solvers import unwrap_phase
from src.visualization import (
    plot_wrapped_phase_and_coherence,
    plot_unwrapped_comparison,
    plot_residuals,
    plot_difference_map,
    compute_metrics,
)

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
EVAL_DIR   = os.path.join(os.path.dirname(__file__), "evaluation")
OUTPUT_DIR = os.path.join(EVAL_DIR, "reference_outputs")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # --- Load data ---
    print("Loading data...")
    data = load_data(os.path.join(DATA_DIR, "raw_data.npz"))
    meta = load_metadata(os.path.join(DATA_DIR, "meta_data.json"))

    interferogram = data["interferogram"]
    wrapped_phase = data["wrapped_phase"]

    # SNAPHU reference lives in data/baseline_reference.npz (moved from raw_data.npz)
    ref_data = np.load(os.path.join(DATA_DIR, "baseline_reference.npz"))
    snaphu_phase = ref_data["unwrapped_phase"][0].astype(np.float64)

    _, coherence = extract_phase_and_coherence(interferogram)

    print(f"  Image size: {wrapped_phase.shape}")
    print(f"  Sensor: {meta['sensor']}, dates: {meta['dates']}")

    # --- Run SPURS unwrapping ---
    _ADMM_PARAMS = {
        "max_iters": 500,
        "tol": np.pi / 5,
        "lmbda": 1,
        "p": 0,
        "c": 1.3,
        "dtype": "float32",
    }
    params = _ADMM_PARAMS
    print(f"\nRunning SPURS unwrapping (max_iters={params['max_iters']}, tol={params['tol']:.4f})...")

    t0 = time.time()
    unwrapped, n_iters = unwrap_phase(
        wrapped_phase,
        max_iters=params["max_iters"],
        tol=params["tol"],
        lmbda=params["lmbda"],
        p=params["p"],
        c=params["c"],
        dtype=params["dtype"],
        debug=True,
    )
    elapsed = time.time() - t0
    print(f"Converged in {n_iters} iterations, {elapsed:.2f} seconds")

    # --- Compute metrics ---
    print("\nComputing metrics...")
    metrics = compute_metrics(unwrapped, snaphu_phase)
    metrics["n_iterations"] = n_iters
    metrics["runtime_seconds"] = round(elapsed, 3)
    metrics["image_shape"] = list(wrapped_phase.shape)

    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # --- Save outputs ---
    print("\nSaving outputs...")
    np.savez(
        os.path.join(OUTPUT_DIR, "unwrapped_phase.npz"),
        unwrapped_spurs=unwrapped,
        unwrapped_snaphu=snaphu_phase,
        wrapped_phase=wrapped_phase,
    )

    with open(os.path.join(EVAL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Also save detailed run metrics to reference_outputs/ for notebook use
    detail_metrics = dict(metrics)
    detail_metrics["n_iterations"] = n_iters
    detail_metrics["runtime_seconds"] = round(elapsed, 3)
    detail_metrics["image_shape"] = list(wrapped_phase.shape)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(detail_metrics, f, indent=2)

    # --- Generate figures ---
    print("Generating figures...")
    plot_wrapped_phase_and_coherence(
        wrapped_phase, coherence,
        save_path=os.path.join(OUTPUT_DIR, "fig1_wrapped_phase_coherence.png"))

    plot_unwrapped_comparison(
        unwrapped, snaphu_phase,
        save_path=os.path.join(OUTPUT_DIR, "fig2_unwrapped_comparison.png"))

    plot_residuals(
        unwrapped, snaphu_phase, wrapped_phase,
        save_path=os.path.join(OUTPUT_DIR, "fig3_residuals.png"))

    plot_difference_map(
        unwrapped, snaphu_phase,
        save_path=os.path.join(OUTPUT_DIR, "fig4_spurs_vs_snaphu_diff.png"))

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
