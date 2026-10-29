"""
Conventional ptychography (CP) reconstruction pipeline.

Loads simulated CP data and runs mPIE reconstruction (7 rounds × 50 iterations).
Saves reconstructed object, probe and metrics to evaluation/reference_outputs/.

Usage:
    python main.py                   # use default data/ directory
    python main.py --data <dir>      # use custom data directory
    python main.py --generate-data   # regenerate simulation data first
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; notebooks use %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TASK_DIR = Path(__file__).parent
DATA_DIR = TASK_DIR / "data"
REFERENCE_DIR = TASK_DIR / "evaluation" / "reference_outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="CP ptychography reconstruction")
    parser.add_argument("--data", type=str, default=str(DATA_DIR),
                        help="Path to data directory containing raw_data.npz and meta_data.json")
    parser.add_argument("--generate-data", action="store_true",
                        help="Regenerate simulation data before reconstruction")
    parser.add_argument("--iterations", type=int, default=350,
                        help="Total mPIE iterations (default 350 = 7×50)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 0. Optionally regenerate simulation data
    # ------------------------------------------------------------------ #
    data_dir = Path(args.data)
    if args.generate_data or not (data_dir / "raw_data.npz").exists():
        logger.info("Generating simulation data …")
        from src.generate_data import main as gen_main
        gen_main(output_path=data_dir / "raw_data.npz")

    # ------------------------------------------------------------------ #
    # 1. Load data and set up reconstruction state
    # ------------------------------------------------------------------ #
    logger.info("Loading data from %s", data_dir)
    from src.preprocessing import (
        load_experimental_data, setup_reconstruction,
        setup_params, setup_monitor, save_results,
    )
    from src.solvers import run_mpie

    data = load_experimental_data(data_dir)
    params = setup_params()
    monitor = setup_monitor(figure_update_freq=50)
    state = setup_reconstruction(data, no_scale=1.0, seed=42)

    logger.info("Reconstruction grid: No=%d, Np=%d, wavelength=%.2f nm, zo=%.1f mm",
                state.No, state.Np,
                state.wavelength * 1e9, state.zo * 1e3)

    # ------------------------------------------------------------------ #
    # 2. Run mPIE reconstruction (7 rounds × 50 iters)
    # ------------------------------------------------------------------ #
    iterations_per_round = 50
    num_rounds = max(1, args.iterations // iterations_per_round)
    logger.info("Running mPIE: %d rounds × %d iterations = %d total",
                num_rounds, iterations_per_round, num_rounds * iterations_per_round)

    params.l2reg_probe_aleph = 1e-2
    params.l2reg_object_aleph = 1e-2

    state = run_mpie(
        state, data, params, monitor,
        num_iterations=num_rounds * iterations_per_round,
        beta_probe=0.25,
        beta_object=0.25,
        iterations_per_round=iterations_per_round,
        seed=42,
    )

    # ------------------------------------------------------------------ #
    # 3. Save reference outputs
    # ------------------------------------------------------------------ #
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    recon_path = REFERENCE_DIR / "recon.hdf5"
    save_results(state, recon_path)
    logger.info("Saved reconstruction to %s", recon_path)

    error = float(state.error[-1]) if state.error else None

    # Compute phase NCC and NRMSE vs ground truth
    gt_path = TASK_DIR / "data" / "ground_truth.npz"
    phase_ncc, phase_nrmse = None, None
    if gt_path.exists():
        gt = np.load(gt_path)["object"]
        obj_arr = state.object
        gt_ph = np.angle(gt)
        obj_ph = np.angle(obj_arr)
        obj_ph -= obj_ph.mean()
        gt_ph_c = gt_ph - gt_ph.mean()
        phase_ncc = float(np.sum(obj_ph * gt_ph_c) / (np.linalg.norm(obj_ph) * np.linalg.norm(gt_ph_c) + 1e-10))
        phase_nrmse = float(np.sqrt(np.mean((obj_ph - gt_ph_c)**2)) / (gt_ph_c.max() - gt_ph_c.min() + 1e-10))
        logger.info("Phase NCC=%.4f, NRMSE=%.4f", phase_ncc, phase_nrmse)

    metrics = {
        "final_error": error,
        "num_iterations": num_rounds * iterations_per_round,
        "object_shape": list(state.object.shape),
        "probe_shape": list(state.probe.shape),
        "wavelength_nm": float(state.wavelength * 1e9),
        "zo_mm": float(state.zo * 1e3),
        "phase_ncc": round(phase_ncc, 4) if phase_ncc is not None else None,
        "phase_nrmse": round(phase_nrmse, 4) if phase_nrmse is not None else None,
    }
    metrics_path = REFERENCE_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics: %s", metrics)

    # PASS/FAIL evaluation against boundaries
    eval_metrics_path = TASK_DIR / "evaluation" / "metrics.json"
    if eval_metrics_path.exists() and phase_ncc is not None:
        with open(eval_metrics_path) as f:
            eval_cfg = json.load(f)
        ncc_ok = phase_ncc >= eval_cfg.get("ncc_boundary", 0.0)
        nrmse_ok = phase_nrmse <= eval_cfg.get("nrmse_boundary", float("inf"))
        status = "PASS" if (ncc_ok and nrmse_ok) else "FAIL"
        logger.info("Evaluation: phase_ncc=%.4f (boundary %.4f) → %s",
                    phase_ncc, eval_cfg.get("ncc_boundary", 0.0), "OK" if ncc_ok else "FAIL")
        logger.info("Evaluation: phase_nrmse=%.4f (boundary %.4f) → %s",
                    phase_nrmse, eval_cfg.get("nrmse_boundary", float("inf")), "OK" if nrmse_ok else "FAIL")
        logger.info("Overall: %s", status)

    # ------------------------------------------------------------------ #
    # 4. Save summary figure
    # ------------------------------------------------------------------ #
    from src.visualization import plot_reconstruction_summary
    error_hist = list(state.error)
    sample_diff = data.ptychogram[0]

    fig = plot_reconstruction_summary(
        state.object, state.probe, error_hist, sample_diff,
        pixel_size_um=float(state.dxo * 1e6),
    )
    fig_path = REFERENCE_DIR / "reconstruction_summary.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved summary figure to %s", fig_path)

    logger.info("Done. Final reconstruction error: %s", error)
    return state, data


if __name__ == "__main__":
    main()
