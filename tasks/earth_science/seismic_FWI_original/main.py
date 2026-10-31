"""Full Waveform Inversion on Marmousi using scratch C-PML wave propagation.

Usage:
    python main.py [--epochs N] [--device cuda] [--output-dir results/]
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

from src.preprocessing import make_initial_model
from src.physics_model import (
    make_acquisition_geometry,
    make_ricker_wavelet,
    forward_model,
)
from src.solvers import run_fwi
from src.visualization import (
    plot_velocity_models,
    plot_shot_gather,
    plot_loss_curve,
    plot_data_comparison,
    compute_data_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="Seismic FWI (scratch C-PML).")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--data", type=str, default="data/raw_data.npz")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    print(f"Loading data from {args.data} ...")
    data = np.load(args.data)
    gt_data = np.load("data/ground_truth.npz")
    v_true = torch.from_numpy(gt_data["v_true"])
    v_init_np = data.get("v_init", None)
    if v_init_np is None:
        v_init = make_initial_model(v_true, sigma=25.0)
    else:
        v_init = torch.from_numpy(v_init_np)
    observed_data = torch.from_numpy(data["observed_data"])
    dx = float(data["dx"])
    freq = float(data["freq"]) if "freq" in data else 5.0
    dt = float(data["dt"]) if "dt" in data else 4e-3
    spacing = (dx, dx)
    ny, nx = v_true.shape
    nt = observed_data.shape[2]
    n_shots, n_receivers = observed_data.shape[0], observed_data.shape[1]

    print(f"  Model: {ny}x{nx} at {dx} m, {n_shots} shots, {n_receivers} receivers, {nt} time steps")
    print(f"  Device: {device}")

    # ------------------------------------------------------------------ #
    # Acquisition geometry
    # ------------------------------------------------------------------ #
    source_loc, receiver_loc = make_acquisition_geometry(
        ny, n_shots=n_shots, n_receivers=n_receivers, device=device
        # ny is the horizontal dimension (first axis); nx is depth
    )
    source_amp = make_ricker_wavelet(freq, nt, dt, n_shots, device=device)
    observed_data = observed_data.to(device)

    # ------------------------------------------------------------------ #
    # Run FWI
    # ------------------------------------------------------------------ #
    print(f"\nStarting FWI: {args.epochs} epochs ...")
    t0 = time.time()
    v_inv, losses = run_fwi(
        v_init, spacing, dt,
        source_amp, source_loc, receiver_loc,
        observed_data, freq,
        n_epochs=args.epochs,
        device=device,
    )
    elapsed = time.time() - t0
    print(f"\nFWI complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ------------------------------------------------------------------ #
    # Evaluate and save reference outputs
    # ------------------------------------------------------------------ #
    print("\nEvaluating final model ...")
    v_inv_dev = v_inv.to(device)
    with torch.no_grad():
        pred_data = forward_model(
            v_inv_dev, spacing, dt, source_amp, source_loc, receiver_loc, freq
        )
    pred_data_np = pred_data.cpu().numpy()
    obs_np = observed_data.cpu().numpy()
    v_inv_np = v_inv.numpy()

    # Velocity NRMSE (RMS error normalised by dynamic range of true velocity model)
    v_true_np = v_true.numpy()
    velocity_nrmse = float(
        np.sqrt(np.mean((v_inv_np - v_true_np)**2)) / (v_true_np.max() - v_true_np.min() + 1e-12)
    )
    print(f"Velocity NRMSE: {velocity_nrmse:.4f}")

    per_shot_rel_l2 = []
    for i in range(n_shots):
        m = compute_data_metrics(obs_np, pred_data_np, shot_idx=i)
        per_shot_rel_l2.append(m["rel_l2"])
    avg_data_rel_l2 = float(np.mean(per_shot_rel_l2))
    print(f"Data rel. L2 error (mean over shots): {avg_data_rel_l2:.4f}")

    import json
    metrics = {
        "velocity_nrmse": velocity_nrmse,
        "final_loss": losses[-1] if losses else None,
        "data_rel_l2_per_shot": per_shot_rel_l2,
        "data_rel_l2_mean": avg_data_rel_l2,
        "n_epochs": args.epochs,
        "elapsed_seconds": elapsed,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(out_dir / "v_inv.npy", v_inv_np)
    np.save(out_dir / "losses.npy", np.array(losses))
    np.save(out_dir / "pred_data.npy", pred_data_np)

    # ------------------------------------------------------------------ #
    # Save figures
    # ------------------------------------------------------------------ #
    fig = plot_velocity_models(v_true_np, v_init.numpy(), v_inv_np, dx, dx)
    fig.savefig(out_dir / "velocity_models.png", dpi=150, bbox_inches="tight")

    fig = plot_loss_curve(losses)
    fig.savefig(out_dir / "loss_curve.png", dpi=150, bbox_inches="tight")

    fig = plot_shot_gather(obs_np, shot_idx=0, title="Observed")
    fig.savefig(out_dir / "observed_shot0.png", dpi=150, bbox_inches="tight")

    fig = plot_data_comparison(obs_np, pred_data_np, shot_idx=0)
    fig.savefig(out_dir / "data_comparison_shot0.png", dpi=150, bbox_inches="tight")

    import matplotlib.pyplot as plt
    plt.close("all")

    print(f"\nResults saved to {out_dir}/")
    loss_str = f", final_loss={losses[-1]:.4e}" if losses else ""
    print(f"  metrics.json: vel_rel_l2={vel_rel_l2:.4f}{loss_str}")


if __name__ == "__main__":
    main()
