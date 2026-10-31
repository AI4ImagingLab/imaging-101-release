"""
Fourier ptychography (FPM) reconstruction pipeline.

Loads simulated FPM data and runs qNewton reconstruction (200 iterations).
Saves reconstructed object, pupil and metrics to evaluation/reference_outputs/.

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TASK_DIR = Path(__file__).parent
DATA_DIR = TASK_DIR / "data"
REFERENCE_DIR = TASK_DIR / "evaluation" / "reference_outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="FPM reconstruction")
    parser.add_argument("--data", type=str, default=str(DATA_DIR),
                        help="Path to data directory containing raw_data.npz and meta_data.json")
    parser.add_argument("--generate-data", action="store_true",
                        help="Regenerate simulation data before reconstruction")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Total qNewton iterations (default 200)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 0. Optionally regenerate simulation data
    # ------------------------------------------------------------------ #
    data_dir = Path(args.data)
    if args.generate_data or not (data_dir / "raw_data.npz").exists():
        logger.info("Generating FPM simulation data …")
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
    from src.solvers import run_qnewton

    data = load_experimental_data(data_dir)
    params = setup_params()
    monitor = setup_monitor(figure_update_freq=50)

    logger.info("FPM data: %d images of %dx%d, λ=%.0f nm",
                data.ptychogram.shape[0],
                data.ptychogram.shape[1],
                data.ptychogram.shape[2],
                data.wavelength * 1e9)

    # ------------------------------------------------------------------ #
    # 2. Initialize reconstruction
    # ------------------------------------------------------------------ #
    state = setup_reconstruction(data, seed=42)

    # ------------------------------------------------------------------ #
    # 3. Run qNewton reconstruction
    # ------------------------------------------------------------------ #
    logger.info("Object: %dx%d, Probe/pupil: %dx%d",
                state.No, state.No, state.Np, state.Np)
    logger.info("Running qNewton: %d iterations", args.iterations)
    state = run_qnewton(
        state, data, params, monitor,
        num_iterations=args.iterations,
    )

    # ------------------------------------------------------------------ #
    # 4. Save reference outputs
    # ------------------------------------------------------------------ #
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    recon_path = REFERENCE_DIR / "recon.hdf5"
    save_results(state, recon_path)
    logger.info("Saved reconstruction to %s", recon_path)

    error = float(state.error[-1]) if state.error else None

    # Compute phase NCC and NRMSE vs ground truth
    # FPM reconstructs in k-space → convert via ifft2c to real-space image
    gt_path = TASK_DIR / "data" / "ground_truth.npz"
    image_phase_ncc, image_phase_nrmse = None, None
    if gt_path.exists():
        from src.utils import ifft2c
        gt = np.load(gt_path)["object"]
        obj_kspace = state.object
        obj_image = ifft2c(obj_kspace)
        gt_ph = np.angle(gt)
        obj_ph = np.angle(obj_image)
        obj_ph -= obj_ph.mean()
        gt_ph_c = gt_ph - gt_ph.mean()
        image_phase_ncc = float(np.sum(obj_ph * gt_ph_c) /
                                (np.linalg.norm(obj_ph) * np.linalg.norm(gt_ph_c) + 1e-10))
        image_phase_nrmse = float(np.sqrt(np.mean((obj_ph - gt_ph_c) ** 2)) /
                                  (gt_ph_c.max() - gt_ph_c.min() + 1e-10))
        logger.info("Image phase NCC=%.4f, NRMSE=%.4f", image_phase_ncc, image_phase_nrmse)

    metrics = {
        "final_error": error,
        "num_iterations": args.iterations,
        "object_shape": list(state.object.shape),
        "pupil_shape": list(state.probe.shape),
        "wavelength_nm": float(state.wavelength * 1e9),
        "NA": float(data.NA),
        "magnification": float(data.magnification),
        "image_phase_ncc": round(image_phase_ncc, 4) if image_phase_ncc is not None else None,
        "image_phase_nrmse": round(image_phase_nrmse, 4) if image_phase_nrmse is not None else None,
        "num_leds": int(data.ptychogram.shape[0]),
    }
    metrics_path = REFERENCE_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics: %s", metrics)

    # PASS/FAIL evaluation against boundaries
    eval_metrics_path = TASK_DIR / "evaluation" / "metrics.json"
    if eval_metrics_path.exists() and image_phase_ncc is not None:
        with open(eval_metrics_path) as f:
            eval_cfg = json.load(f)
        ncc_ok = image_phase_ncc >= eval_cfg.get("ncc_boundary", 0.0)
        nrmse_ok = image_phase_nrmse <= eval_cfg.get("nrmse_boundary", float("inf"))
        status = "PASS" if (ncc_ok and nrmse_ok) else "FAIL"
        logger.info("Evaluation: image_phase_ncc=%.4f (boundary %.4f) → %s",
                    image_phase_ncc, eval_cfg.get("ncc_boundary", 0.0), "OK" if ncc_ok else "FAIL")
        logger.info("Evaluation: image_phase_nrmse=%.4f (boundary %.4f) → %s",
                    image_phase_nrmse, eval_cfg.get("nrmse_boundary", float("inf")), "OK" if nrmse_ok else "FAIL")
        logger.info("Overall: %s", status)

    # ------------------------------------------------------------------ #
    # 5. Save summary figure
    # ------------------------------------------------------------------ #
    from src.visualization import plot_reconstruction_summary
    error_hist = list(state.error)
    fig = plot_reconstruction_summary(state, data, error_hist)
    fig_path = REFERENCE_DIR / "reconstruction_summary.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved summary figure to %s", fig_path)

    logger.info("Done. Final error: %s", error)
    return state, data


if __name__ == "__main__":
    main()
