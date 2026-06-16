"""Least-Squares Reverse-Time Migration — from-scratch C-PML implementation.

All wave propagation implemented from scratch without deepwave.

Usage:
    python main.py [--epochs N] [--device cuda] [--output-dir results/]
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.physics_model import (
    make_acquisition_geometry,
    make_ricker_wavelet,
    born_forward_model,
)
from src.solvers import subtract_direct_arrival, run_lsrtm
from src.visualization import (
    plot_scatter_image,
    plot_velocity_models,
    plot_scattered_data,
    plot_loss_curve,
    plot_data_comparison,
    compute_data_metrics,
)

_DEFAULT_EPOCHS = 3
_DEFAULT_OUTPUT_DIR = "results"


def _load_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _squeeze_batch(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim == 0 or array.shape[0] != 1:
        raise ValueError(f"{name} must use batch-first shape (1, ...), got {array.shape}")
    return array[0]


def main():
    parser = argparse.ArgumentParser(description="Seismic LSRTM (scratch C-PML)")
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data", type=str, default="data/raw_data.npz")
    parser.add_argument("--metadata", type=str, default="data/meta_data.json")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print(f"Loading data from {args.data} ...")
    data = np.load(args.data)
    gt = np.load("data/ground_truth.npz")
    meta = _load_metadata(args.metadata)

    v_true = torch.from_numpy(_squeeze_batch(gt["v_true"], "ground_truth['v_true']"))
    v_mig = torch.from_numpy(_squeeze_batch(data["v_mig"], "raw_data['v_mig']")).to(device)
    observed_data = torch.from_numpy(
        _squeeze_batch(data["observed_data"], "raw_data['observed_data']")
    ).to(device)
    dx = float(meta["preprocessing"]["dx_m"])
    dt = float(meta["time"]["dt_s"])
    freq = float(meta["wavelet"]["frequency_hz"])
    nt = int(meta["time"]["nt"])
    n_shots = int(meta["acquisition"]["n_shots"])
    n_rec = int(meta["acquisition"]["n_receivers_per_shot"])
    ny, nx = v_true.shape

    print(f"  Model: {ny}x{nx} at {dx} m, {n_shots} shots, {n_rec} receivers, {nt} time steps")
    print(f"  Device: {device}")

    # ── 2. Acquisition geometry ───────────────────────────────────────────
    acq = meta["acquisition"]
    source_loc, receiver_loc = make_acquisition_geometry(
        n_shots=n_shots,
        d_source=int(acq["d_source"]),
        first_source=int(acq["first_source"]),
        source_depth=int(acq["source_depth"]),
        n_receivers=n_rec,
        d_receiver=int(acq["d_receiver"]),
        first_receiver=int(acq["first_receiver"]),
        receiver_depth=int(acq["receiver_depth"]),
        device=device,
    )
    source_amp = make_ricker_wavelet(freq, nt, dt, n_shots, device=device)

    # ── 3. Subtract direct arrivals ───────────────────────────────────────
    print("Subtracting direct arrivals ...")
    scattered_data = subtract_direct_arrival(
        observed_data, v_mig, dx, dt,
        source_amp, source_loc, receiver_loc, freq,
        v_true_max=float(v_true.max()),
    )
    print(f"  Scattered data range: [{scattered_data.min().item():.4e}, {scattered_data.max().item():.4e}]")
    direct_data = observed_data - scattered_data

    # ── 4. Run LSRTM ─────────────────────────────────────────────────────
    print(f"\nStarting LSRTM: {args.epochs} L-BFGS iterations ...")
    t0 = time.time()
    scatter, losses = run_lsrtm(
        v_mig, dx, dt,
        source_amp, source_loc, receiver_loc,
        scattered_data, freq,
        n_epochs=args.epochs,
        device=device,
    )
    elapsed = time.time() - t0
    print(f"\nLSRTM complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    print("\nEvaluating ...")
    with torch.no_grad():
        pred_scattered = born_forward_model(
            v_mig, scatter.to(device), dx, dt,
            source_amp, source_loc, receiver_loc, freq,
        )
    pred_np = pred_scattered.cpu().numpy()
    obs_scat_np = scattered_data.cpu().numpy()
    scatter_np = scatter.numpy()

    per_shot_rel_l2 = []
    for i in range(n_shots):
        m = compute_data_metrics(obs_scat_np, pred_np, shot_idx=i)
        per_shot_rel_l2.append(m["rel_l2"])
    avg_data_rel_l2 = float(np.mean(per_shot_rel_l2))
    print(f"Data rel. L2 error (mean over shots): {avg_data_rel_l2:.4f}")

    metrics = {
        "final_loss": losses[-1] if losses else None,
        "data_rel_l2_per_shot": per_shot_rel_l2,
        "data_rel_l2_mean": avg_data_rel_l2,
        "n_epochs": args.epochs,
        "elapsed_seconds": elapsed,
        "scatter_range": [float(scatter.min()), float(scatter.max())],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(os.path.join(out_dir, "reference_reconstruction.npz"), scatter=scatter_np[np.newaxis])
    np.savez(os.path.join(out_dir, "losses.npz"), losses=np.array(losses))
    np.savez(os.path.join(out_dir, "pred_scattered.npz"), pred_scattered=pred_np)
    np.savez(os.path.join(out_dir, "scattered_data.npz"), scattered_data=obs_scat_np)

    # ── 6. Save figures ───────────────────────────────────────────────────
    fig = plot_scatter_image(scatter_np, dx, title="LSRTM Image (scratch solver)")
    fig.savefig(os.path.join(out_dir, "lsrtm_image.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_velocity_models(v_true.numpy(), v_mig.cpu().numpy(), dx)
    fig.savefig(os.path.join(out_dir, "velocity_models.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_scattered_data(
        observed_data.cpu().numpy(), direct_data.cpu().numpy(),
        obs_scat_np, shot_idx=0,
    )
    fig.savefig(os.path.join(out_dir, "scattered_decomposition.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_loss_curve(losses)
    fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_data_comparison(obs_scat_np, pred_np, shot_idx=0)
    fig.savefig(os.path.join(out_dir, "data_comparison_shot0.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nResults saved to {out_dir}/")
    loss_str = f", final_loss={losses[-1]:.4e}" if losses else ""
    print(f"  metrics.json: data_rel_l2_mean={avg_data_rel_l2:.4f}{loss_str}")


if __name__ == "__main__":
    main()
