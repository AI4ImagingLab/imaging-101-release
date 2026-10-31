"""Generate test fixtures for weather_radar_data_assimilation task."""
import os
import sys
import numpy as np
import torch

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from src.generate_data import generate_observations
from src.physics_model import make_observation_operator
from src.preprocessing import scale_to_model, load_ground_truth
from src.solvers import StochasticInterpolant, DriftModel


DATA_DIR = os.path.join(TASK_DIR, "data")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    fixture_dir = os.path.join(TASK_DIR, "evaluation", "fixtures")
    ensure_dir(fixture_dir)

    # ── generate_data.npz ──────────────────────────────────────────────
    gt = load_ground_truth(DATA_DIR)  # (1, 3, 128, 128) float32
    # Use first sample
    gt_sample = gt[0]  # (3, 128, 128)
    observations, mask = generate_observations(gt_sample, mask_ratio=0.1, noise_sigma=0.001, seed=42)

    np.savez(
        os.path.join(fixture_dir, "generate_data.npz"),
        input_ground_truth=gt_sample,
        output_observations=observations,
        output_mask=mask,
    )
    print("[OK] generate_data.npz")

    # ── physics_model_mask.npz ─────────────────────────────────────────
    np.random.seed(100)
    # Create a small test mask and input
    input_x = np.random.rand(3, 128, 128).astype(np.float32)
    param_mask = (np.random.rand(1, 128, 128) < 0.1).astype(np.float32)
    op = make_observation_operator(param_mask)
    output_masked = op(input_x)

    np.savez(
        os.path.join(fixture_dir, "physics_model_mask.npz"),
        input_x=input_x,
        param_mask=param_mask,
        output_masked=output_masked,
    )
    print("[OK] physics_model_mask.npz")

    # ── preprocessing_scale.npz ────────────────────────────────────────
    np.random.seed(200)
    input_pixels = np.random.rand(3, 128, 128).astype(np.float32)
    output_scaled = scale_to_model(input_pixels)

    np.savez(
        os.path.join(fixture_dir, "preprocessing_scale.npz"),
        input_pixels=input_pixels,
        output_scaled=output_scaled,
    )
    print("[OK] preprocessing_scale.npz")

    # ── solvers_interpolant.npz ────────────────────────────────────────
    interp = StochasticInterpolant(beta_fn="t^2", sigma_coef=1.0)
    t_vals = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=np.float32)
    alphas = []
    betas = []
    sigmas = []
    for t_val in t_vals:
        t_tensor = torch.tensor([t_val])
        alphas.append(interp.alpha(t_tensor).item())
        betas.append(interp.beta(t_tensor).item())
        sigmas.append(interp.sigma(t_tensor).item())

    np.savez(
        os.path.join(fixture_dir, "solvers_interpolant.npz"),
        input_t=t_vals,
        output_alpha=np.array(alphas, dtype=np.float32),
        output_beta=np.array(betas, dtype=np.float32),
        output_sigma=np.array(sigmas, dtype=np.float32),
    )
    print("[OK] solvers_interpolant.npz")

    # ── solvers_drift_model.npz ────────────────────────────────────────
    # Build model to get expected output shape
    model = DriftModel(in_channels=7, out_channels=1, unet_channels=128)
    model.eval()
    zt = torch.randn(1, 1, 128, 128)
    cond = torch.randn(1, 6, 128, 128)
    t = torch.tensor([0.5])
    with torch.no_grad():
        out = model(zt, t, None, cond=cond)

    np.savez(
        os.path.join(fixture_dir, "solvers_drift_model.npz"),
        input_shape=np.array(zt.shape),
        output_shape=np.array(out.shape),
    )
    print("[OK] solvers_drift_model.npz")

    print("\nAll weather_radar_data_assimilation fixtures generated successfully!")


if __name__ == "__main__":
    main()
