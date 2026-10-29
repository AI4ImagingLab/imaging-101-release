"""
USCT Full-Waveform Inversion with CBS Solver
=============================================

Reconstructs the sound speed distribution in biological tissue from
ultrasound transmission measurements using frequency-domain FWI with
a Convergent Born Series (CBS) Helmholtz solver.

Usage:
    cd tasks/usct_FWI
    python main.py                          # Multi-frequency (full pipeline)
    python main.py --mode single --freq 0.3 # Single frequency
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import (
    load_metadata,
    load_observations,
    load_baseline_reference,
    build_restriction_operator,
    create_initial_slowness,
)
from src.solvers import invert_single_frequency, invert_multi_frequency
from src.visualization import (
    compute_ncc,
    compute_nrmse,
    plot_velocity,
    plot_comparison,
    plot_convergence,
)

# Solver hyperparameters (not in meta_data.json per imaging-101 convention)
_NCG_ITERS = 3
_LINESEARCH_MAXFEV = 5
_SIGMA_SCHEDULE = {0.3: 5, 0.8: 2, 0.8001: 1}  # freq thresholds -> sigma


def main():
    parser = argparse.ArgumentParser(description="USCT FWI with CBS solver")
    parser.add_argument("--mode", default="multi", choices=["single", "multi"])
    parser.add_argument("--freq", type=float, default=0.3, help="Frequency for single mode (MHz)")
    parser.add_argument("--device", default="0", help="CUDA device")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.cuda.set_device(f"cuda:{args.device}")

    data_dir = os.path.join(TASK_DIR, "data")
    output_dir = os.path.join(TASK_DIR, "evaluation", "reference_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 1: Load data and metadata
    # ---------------------------------------------------------------
    meta = load_metadata(data_dir)
    obs = load_observations(data_dir)
    ix, iy = obs["receiver_ix"], obs["receiver_iy"]

    vp_ref = load_baseline_reference(data_dir)
    logging.info(f"Grid: {meta['nx']}x{meta['ny']}, {meta['n_transducers_active']} transducers")

    # ---------------------------------------------------------------
    # Step 2: Build restriction operator and initial model
    # ---------------------------------------------------------------
    R = build_restriction_operator(ix, iy, meta["nx"], meta["ny"])
    slowness = create_initial_slowness(meta["nx"], meta["ny"], meta["background_velocity_mps"])
    all_u_size = [meta["nx"], meta["ny"], meta["n_transducers_active"]]

    cbs_kwargs = dict(
        dh=meta["dh_um"],
        ppw=meta["cbs_ppw"],
        lamb=meta["cbs_lamb"],
        boundary_widths=meta["cbs_boundary_widths"],
        born_max=meta["cbs_born_max"],
        energy_threshold=meta["cbs_energy_threshold"],
    )

    # ---------------------------------------------------------------
    # Step 3: Run inversion
    # ---------------------------------------------------------------
    if args.mode == "single":
        freq_str = f"{args.freq:g}"
        dobs = obs["dobs_all"][freq_str]
        sigma = 5 if args.freq <= 0.3 else (2 if args.freq < 0.8 else 1)

        logging.info(f"Running single-frequency FWI at {args.freq} MHz, sigma={sigma}")
        slowness = invert_single_frequency(
            args.freq, dobs, sigma, slowness, ix, iy, R, all_u_size,
            ncg_iters=_NCG_ITERS, v_bounds=tuple(meta["velocity_bounds_mps"]),
            **cbs_kwargs,
        )
        vp_recon = (1 / slowness).cpu().numpy()
        history = [(args.freq, vp_recon)]

    else:  # multi-frequency
        freqs = meta["frequencies_MHz"]
        logging.info(f"Running multi-frequency FWI: {len(freqs)} frequencies")
        slowness, history = invert_multi_frequency(
            freqs, obs["dobs_all"], slowness, ix, iy, R, all_u_size,
            ncg_iters=_NCG_ITERS, v_bounds=tuple(meta["velocity_bounds_mps"]),
            **cbs_kwargs,
        )
        vp_recon = (1 / slowness).cpu().numpy()

    # ---------------------------------------------------------------
    # Step 4: Compute metrics
    # ---------------------------------------------------------------
    ncc = compute_ncc(vp_recon, vp_ref)
    nrmse = compute_nrmse(vp_recon, vp_ref)
    logging.info(f"Final metrics: NCC={ncc:.4f}, NRMSE={nrmse:.4f}")

    # ---------------------------------------------------------------
    # Step 5: Save outputs
    # ---------------------------------------------------------------
    if args.mode == "single":
        # Single-freq outputs use freq suffix to avoid overwriting multi-freq results
        freq_str = f"{args.freq:g}"
        np.save(os.path.join(output_dir, f"reconstruction_{freq_str}.npy"), vp_recon)
        plot_velocity(vp_recon, title=f"Reconstructed Sound Speed ({freq_str} MHz)",
                      save_path=os.path.join(output_dir, f"reconstruction_{freq_str}.png"))
    else:
        # Multi-freq outputs use standard names (evaluation harness expects these)
        np.save(os.path.join(output_dir, "reconstruction.npy"), vp_recon)
        np.save(os.path.join(output_dir, "baseline_reference.npy"), vp_ref)

        metrics = {
            "ncc_vs_ref": round(ncc, 4),
            "nrmse_vs_ref": round(nrmse, 4),
            "mode": "multi",
            "freq": "all",
        }
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save per-frequency intermediate results
        for freq_mhz, vp_hist in history:
            fs = f"{freq_mhz:g}"
            np.save(os.path.join(output_dir, f"vp_freq_{fs}.npy"), vp_hist)

        # ---------------------------------------------------------------
        # Step 6: Visualizations (multi-freq only)
        # ---------------------------------------------------------------
        plot_velocity(vp_recon, title="Reconstructed Sound Speed (Multi-freq)",
                      save_path=os.path.join(output_dir, "reconstruction.png"))
        plot_comparison(vp_ref, vp_recon,
                        save_path=os.path.join(output_dir, "comparison.png"))
        plot_convergence(history, vp_ref,
                         save_path=os.path.join(output_dir, "convergence.png"))

    logging.info(f"Outputs saved to {output_dir}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
