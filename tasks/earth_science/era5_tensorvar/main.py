"""End-to-end Tensor-Var 4D-Var pipeline for the bundled ERA5 sample.

Pipeline
--------
1. Ensure the four pretrained checkpoints are present (download from the
   upstream Google Drive release if missing).
2. Load the bundled single-sample observation history and ground truth.
3. Run Tensor-Var 4D-Var (one assimilation window of length 5).
4. Save the analysis trajectory, per-channel metrics, comparison figures and
   QP intermediates to `evaluation/reference_outputs/`.

All solver hyperparameters live as named constants below; nothing is read from
`data/meta_data.json` except imaging geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np
import torch

from src.preprocessing import (
    default_covariances,
    download_pretrained_weights,
    load_ground_truth,
    load_metadata,
    load_observation,
    load_pretrained_models,
    select_sample,
)
from src.solvers import tensor_var_4dvar
from src.visualization import (
    compute_metrics_per_channel,
    metrics_to_jsonable,
    plot_all_channels,
    print_metrics_table,
)


# ---------------------------------------------------------------------------
# Hyperparameters (match the parity capture; do NOT move to meta_data.json).
# ---------------------------------------------------------------------------
_ASS_W = 5         # one assimilation window of length 5
_ASS_T = 5         # total timesteps assimilated == one window
_SEED = 0          # numpy / torch RNG seed
_SAMPLE_INDEX = 0  # batch index inside raw_data.npz / ground_truth.npz


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = TASK_DIR / "data"
DEFAULT_WEIGHTS_DIR = TASK_DIR / "evaluation" / "checkpoints"
DEFAULT_OUTPUT_DIR = TASK_DIR / "evaluation" / "reference_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ERA5 Tensor-Var 4D-Var pipeline")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS_DIR))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download pretrained weights and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)

    weights_dir = Path(args.weights)
    download_pretrained_weights(weights_dir)
    if args.download_only:
        return

    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Step 1/5  Loading bundled sample from {data_dir}")
    raw = select_sample(load_observation(data_dir), _SAMPLE_INDEX)
    gt = select_sample(load_ground_truth(data_dir), _SAMPLE_INDEX)
    meta = load_metadata(data_dir)
    obs_history = torch.tensor(raw["obs_history"], dtype=torch.float32, device=args.device)
    state_true = gt["state"]                          # (T, C, H, W)  normalised
    lat_weight_matrix = raw["lat_weight_matrix"]      # (C, H, W)

    print("[INFO] Step 2/5  Loading pretrained Tensor-Var checkpoints")
    forward_model, inverse_model, z_b = load_pretrained_models(weights_dir, args.device)

    B, R, Q = default_covariances(forward_model.hidden_dim)

    print("[INFO] Step 3/5  Running Tensor-Var 4D-Var")
    trajectory, diagnostics = tensor_var_4dvar(
        obs_history=obs_history,
        forward_model=forward_model,
        inverse_model=inverse_model,
        z_b=z_b,
        B=B,
        R=R,
        Q=Q,
        assimilation_window=_ASS_W,
        total_steps=_ASS_T,
    )
    print(f"[INFO]   QP eval time: {diagnostics['evaluation_time_s']:.2f} s")

    print("[INFO] Step 4/5  Computing per-channel metrics")
    metrics = compute_metrics_per_channel(trajectory, state_true, lat_weight_matrix)
    print_metrics_table(metrics)

    print(f"[INFO] Step 5/5  Saving artefacts to {output_dir}")
    np.save(output_dir / "trajectory.npy", trajectory)
    np.save(output_dir / "ground_truth.npy", state_true)
    np.save(output_dir / "inv_obs_seq_z.npy", diagnostics["inv_obs_seq_z"])
    np.save(output_dir / "K_S_seq_z.npy", diagnostics["K_S_seq_z"])
    np.save(output_dir / "qp_result.npy", diagnostics["qp_result"])

    metrics_record = metrics_to_jsonable(metrics)
    metrics_record["channels"] = meta["channels"]
    metrics_record["evaluation_time_s"] = round(float(diagnostics["evaluation_time_s"]), 4)
    with (output_dir / "metrics.json").open("w") as fh:
        json.dump(metrics_record, fh, indent=2)

    fig = plot_all_channels(trajectory, state_true, metrics)
    fig.savefig(output_dir / "comparison.png", dpi=120, bbox_inches="tight")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
