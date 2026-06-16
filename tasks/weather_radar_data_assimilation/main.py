"""Weather Radar Data Assimilation — reconstruction pipeline.

Reconstructs full-resolution future VIL radar frames from sparse point
observations and past context using FlowDAS (Chen et al., 2025): a flow-based
data assimilation method with guided Euler-Maruyama sampling.
"""
import matplotlib
matplotlib.use("Agg")

import os
import json
import numpy as np
import torch

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "data")
OUTPUT_DIR = os.path.join(TASK_DIR, "output")
EVAL_DIR = os.path.join(TASK_DIR, "evaluation")
REF_DIR = os.path.join(EVAL_DIR, "reference_outputs")

# ── Solver parameters (not leaked through meta_data.json) ───────────────────
_SOLVER_PARAMS = {
    "n_steps": 500,
    "mc_times": 25,
    "guidance_scale": 0.1,
    "sigma_coef": 1.0,
    "beta_fn": "t^2",
}

# ── Checkpoint path ─────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(TASK_DIR, "model", "latest.pt")


def compute_ncc(x, ref):
    """Cosine similarity between flattened vectors (no mean subtraction)."""
    x_flat = x.flatten()
    r_flat = ref.flatten()
    return float(np.dot(x_flat, r_flat) / (np.linalg.norm(x_flat) * np.linalg.norm(r_flat)))


def compute_nrmse(x, ref):
    """NRMSE normalised by dynamic range of reference."""
    rms = np.sqrt(np.mean((x - ref) ** 2))
    dyn_range = ref.max() - ref.min()
    return float(rms / dyn_range)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────
    from src.preprocessing import load_raw_data, load_ground_truth, load_meta_data, scale_to_model, scale_from_model

    raw = load_raw_data(DATA_DIR)
    gt_data = load_ground_truth(DATA_DIR)
    meta = load_meta_data(DATA_DIR)

    condition_frames = raw["condition_frames"][0]   # (6, 128, 128)
    observations = raw["observations"][0]           # (3, 128, 128)
    mask = raw["observation_mask"][0]               # (1, 128, 128)
    target_frames = gt_data[0]                      # (3, 128, 128)

    print(f"Condition: {condition_frames.shape}, Observations: {observations.shape}")
    print(f"Mask coverage: {mask.mean():.4f}, Target: {target_frames.shape}")

    # ── 2. Set up forward model ─────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    from src.physics_model import make_observation_operator, make_noiser

    mask_tensor = torch.from_numpy(mask).float().to(device)
    operator = make_observation_operator(mask_tensor)
    noiser = make_noiser(meta["noise_sigma"])

    # ── 3. Scale data to model space ────────────────────────────────────
    cond_scaled = scale_to_model(condition_frames)
    obs_scaled = scale_to_model(observations)

    cond_tensor = torch.from_numpy(cond_scaled).float().unsqueeze(0).to(device)  # (1, 6, 128, 128)
    obs_tensor = torch.from_numpy(obs_scaled).float().unsqueeze(0).to(device)    # (1, 3, 128, 128)

    # ── 4. Load model and run reconstruction ────────────────────────────
    from src.solvers import load_drift_model, StochasticInterpolant, autoregressive_reconstruct

    model = load_drift_model(CHECKPOINT_PATH, device=device)
    interpolant = StochasticInterpolant(
        beta_fn=_SOLVER_PARAMS["beta_fn"],
        sigma_coef=_SOLVER_PARAMS["sigma_coef"],
    )

    with torch.no_grad():
        recon_tensor = autoregressive_reconstruct(
            model, interpolant, cond_tensor, obs_tensor,
            operator, noiser,
            n_steps=_SOLVER_PARAMS["n_steps"],
            mc_times=_SOLVER_PARAMS["mc_times"],
            auto_steps=meta["n_prediction_frames"],
            device=device,
        )

    # ── 5. Convert back to pixel space ──────────────────────────────────
    recon = scale_from_model(recon_tensor[0].cpu().numpy())  # (3, 128, 128)
    print(f"Reconstruction range: [{recon.min():.4f}, {recon.max():.4f}]")

    # ── 6. Evaluate ─────────────────────────────────────────────────────
    ncc_val = compute_ncc(recon, target_frames)
    nrmse_val = compute_nrmse(recon, target_frames)
    print(f"NCC vs ground truth:   {ncc_val:.4f}")
    print(f"NRMSE vs ground truth: {nrmse_val:.4f}")

    # Per-frame metrics
    for i in range(target_frames.shape[0]):
        ncc_i = compute_ncc(recon[i], target_frames[i])
        nrmse_i = compute_nrmse(recon[i], target_frames[i])
        print(f"  Frame {i}: NCC={ncc_i:.4f}, NRMSE={nrmse_i:.4f}")

    # Save reconstruction
    np.savez(
        os.path.join(OUTPUT_DIR, "reconstruction.npz"),
        reconstructed_frames=recon[np.newaxis].astype(np.float32),
    )

    # Save metrics
    metrics = {
        "baseline": [
            {"method": "FlowDAS (MC=25, 500 EM steps)", "ncc_vs_ref": ncc_val, "nrmse_vs_ref": nrmse_val}
        ],
        "ncc_boundary": round(0.9 * ncc_val, 4),
        "nrmse_boundary": round(1.1 * nrmse_val, 4),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── 7. Visualize ────────────────────────────────────────────────────
    from src.visualization import plot_comparison, plot_error_map

    plot_comparison(
        condition_frames, target_frames, observations, recon,
        save_path=os.path.join(OUTPUT_DIR, "comparison.png"),
        mask=mask,
    )
    plot_error_map(
        target_frames, recon,
        save_path=os.path.join(OUTPUT_DIR, "error_map.png"),
    )

    print("Done. Results saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
