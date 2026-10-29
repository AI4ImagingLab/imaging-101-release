"""Entry point for X-ray ptychography reconstruction.

Loads data, runs iterative ePIE reconstruction with least-squares step sizes,
computes evaluation metrics, and saves outputs to the output/ directory.
"""

import matplotlib
matplotlib.use("Agg")

import json
import logging
import os
import sys

import numpy as np

# Add task root to path so src imports work
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

from src.preprocessing import (
    load_raw_data,
    load_metadata,
    shift_scan_positions,
    initialize_psi,
    add_probe_modes,
)
from src.solvers import reconstruct
from src.visualization import (
    compute_metrics,
    plot_phase,
    plot_cost_curve,
)


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    data_dir = os.path.join(TASK_DIR, "data")
    output_dir = os.path.join(TASK_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load data ----
    logger.info("Loading raw data...")
    raw = load_raw_data(data_dir)
    meta = load_metadata(data_dir)

    data = raw["diffraction_patterns"]
    scan = raw["scan_positions"]
    probe = raw["probe_guess"]

    logger.info(f"Diffraction patterns: {data.shape}, dtype={data.dtype}")
    logger.info(f"Scan positions: {scan.shape}, dtype={scan.dtype}")
    logger.info(f"Probe guess: {probe.shape}, dtype={probe.dtype}")

    # ---- Preprocessing ----
    logger.info("Preprocessing scan positions and initializing arrays...")
    scan = shift_scan_positions(scan, offset=20.0)
    probe = add_probe_modes(probe, n_modes=1)
    psi = initialize_psi(scan, probe_shape=probe.shape)

    logger.info(f"Probe (after mode init): {probe.shape}")
    logger.info(f"Object (psi): {psi.shape}")

    # ---- Reconstruction ----
    logger.info("Starting reconstruction...")
    result = reconstruct(
        data=data,
        scan=scan,
        probe=probe,
        psi=psi,
        num_iter=64,
        num_batch=7,
    )

    # ---- Extract results ----
    recon_psi = result['psi']
    recon_probe = result['probe']
    costs = np.array(result['costs'])

    logger.info(f"Reconstructed object shape: {recon_psi.shape}")
    logger.info(f"Reconstructed probe shape: {recon_probe.shape}")
    logger.info(f"Final cost: {costs[-1].mean():.4f}")

    # ---- Save outputs ----
    np.save(os.path.join(output_dir, "reconstructed_object.npy"), recon_psi)
    np.save(os.path.join(output_dir, "reconstructed_probe.npy"), recon_probe)
    np.save(os.path.join(output_dir, "costs.npy"), costs)

    # ---- Visualization ----
    logger.info("Generating plots...")
    plot_phase(
        recon_psi[0],
        title="Reconstructed Object Phase",
        save_path=os.path.join(output_dir, "reconstructed_object.png"),
    )
    plot_cost_curve(
        costs,
        title="Reconstruction Cost Curve",
        save_path=os.path.join(output_dir, "cost_curve.png"),
    )

    # ---- Metrics ----
    baseline_path = os.path.join(data_dir, "baseline_reference.npz")
    if os.path.exists(baseline_path):
        logger.info("Computing metrics against baseline reference...")
        baseline = np.load(baseline_path)
        ref_phase = baseline["object_phase"]

        # Extract phase from reconstruction (squeeze depth dim to match)
        est_phase = np.angle(recon_psi)

        # Crop to matching shape (reference may be a different size)
        min_h = min(est_phase.shape[-2], ref_phase.shape[-2])
        min_w = min(est_phase.shape[-1], ref_phase.shape[-1])
        est_crop = est_phase[:, :min_h, :min_w]
        ref_crop = ref_phase[:, :min_h, :min_w]

        metrics = compute_metrics(est_crop, ref_crop)
        metrics["final_cost"] = float(costs[-1].mean())
        logger.info(f"Metrics: NCC={metrics['ncc']:.4f}, "
                     f"NRMSE={metrics['nrmse']:.4f}")
    else:
        logger.warning("No baseline_reference.npz found, skipping metrics.")
        metrics = {"final_cost": float(costs[-1].mean())}

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
