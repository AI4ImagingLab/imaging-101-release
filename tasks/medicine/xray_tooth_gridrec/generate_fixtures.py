"""Generate test fixtures for xray_tooth_gridrec task."""
import os
import sys
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from src.physics_model import ParallelBeamProjector, find_rotation_center
from src.preprocessing import load_observation, normalize, minus_log
from src.solvers import ramp_filter, filtered_back_projection, circular_mask
from src.visualization import compute_metrics


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    # ── physics_model fixtures ──────────────────────────────────────────
    pm_dir = os.path.join(TASK_DIR, "evaluation", "fixtures", "physics_model")
    ensure_dir(pm_dir)

    # projector.npz
    n_pixels = 16
    n_detector = 16
    theta = np.linspace(0, np.pi, 18, endpoint=False)
    projector = ParallelBeamProjector(n_pixels, n_detector, theta)

    np.random.seed(42)
    image = np.random.rand(n_pixels, n_pixels)
    sinogram = projector.forward(image)
    adjoint = projector.adjoint(sinogram)

    np.savez(
        os.path.join(pm_dir, "projector.npz"),
        param_n_pixels=np.array(n_pixels),
        param_n_detector=np.array(n_detector),
        param_theta=theta,
        input_image=image,
        output_sinogram=sinogram,
        output_adjoint=adjoint,
    )
    print("[OK] physics_model/projector.npz")

    # rotation_center.npz - use real data
    data_dir = os.path.join(TASK_DIR, "data")
    obs = load_observation(data_dir)
    proj = obs["projections"]
    flat = obs["flat_field"]
    dark = obs["dark_field"]
    theta_real = obs["theta"]

    norm_proj = normalize(proj, flat, dark)
    sino_real = minus_log(norm_proj)
    # Pick single slice (index 0)
    sino_slice = sino_real[:, 0, :]
    init_center = sino_slice.shape[1] / 2.0
    tol = 0.5
    center = find_rotation_center(sino_slice, theta_real, init=init_center, tol=tol)

    np.savez(
        os.path.join(pm_dir, "rotation_center.npz"),
        input_sinogram=sino_slice,
        input_theta=theta_real,
        config_init=np.array(init_center),
        config_tol=np.array(tol),
        output_center=np.array(center),
    )
    print("[OK] physics_model/rotation_center.npz")

    # ── preprocessing fixtures ──────────────────────────────────────────
    pp_dir = os.path.join(TASK_DIR, "evaluation", "fixtures", "preprocessing")
    ensure_dir(pp_dir)

    # normalize.npz - small random arrays
    np.random.seed(123)
    input_proj = np.random.rand(5, 2, 10) * 1000
    input_flat = np.random.rand(3, 2, 10) * 1200
    input_dark = np.random.rand(3, 2, 10) * 50
    output_normalized = normalize(input_proj, input_flat, input_dark)

    np.savez(
        os.path.join(pp_dir, "normalize.npz"),
        input_proj=input_proj,
        input_flat=input_flat,
        input_dark=input_dark,
        output_normalized=output_normalized,
    )
    print("[OK] preprocessing/normalize.npz")

    # minus_log.npz
    np.random.seed(456)
    input_normalized = np.random.rand(5, 10) * 0.9 + 0.05  # in (0.05, 0.95)
    output_sinogram = minus_log(input_normalized)

    np.savez(
        os.path.join(pp_dir, "minus_log.npz"),
        input_normalized=input_normalized,
        output_sinogram=output_sinogram,
    )
    print("[OK] preprocessing/minus_log.npz")

    # ── solvers fixtures ────────────────────────────────────────────────
    sol_dir = os.path.join(TASK_DIR, "evaluation", "fixtures", "solvers")
    ensure_dir(sol_dir)

    # ramp_filter.npz (test uses n_detector=16)
    filt = ramp_filter(16)
    np.savez(
        os.path.join(sol_dir, "ramp_filter.npz"),
        output_filter=filt,
    )
    print("[OK] solvers/ramp_filter.npz")

    # fbp.npz
    np.random.seed(789)
    fbp_n_pixels = 16
    fbp_theta = np.linspace(0, np.pi, 18, endpoint=False)
    # Create a small sinogram
    fbp_projector = ParallelBeamProjector(fbp_n_pixels, fbp_n_pixels, fbp_theta)
    fbp_image = np.random.rand(fbp_n_pixels, fbp_n_pixels)
    fbp_sinogram = fbp_projector.forward(fbp_image)
    fbp_recon = filtered_back_projection(fbp_sinogram, fbp_theta, fbp_n_pixels)

    np.savez(
        os.path.join(sol_dir, "fbp.npz"),
        input_sinogram=fbp_sinogram,
        input_theta=fbp_theta,
        param_n_pixels=np.array(fbp_n_pixels),
        output_reconstruction=fbp_recon,
    )
    print("[OK] solvers/fbp.npz")

    # circular_mask.npz
    np.random.seed(101)
    mask_image = np.random.rand(32, 32)
    mask_ratio = 0.85
    masked_output = circular_mask(mask_image, ratio=mask_ratio)

    np.savez(
        os.path.join(sol_dir, "circular_mask.npz"),
        input_image=mask_image,
        config_ratio=np.array(mask_ratio),
        output_masked=masked_output,
    )
    print("[OK] solvers/circular_mask.npz")

    # ── visualization fixtures ──────────────────────────────────────────
    vis_dir = os.path.join(TASK_DIR, "evaluation", "fixtures", "visualization")
    ensure_dir(vis_dir)

    np.random.seed(202)
    input_estimate = np.random.rand(16, 16)
    input_reference = np.random.rand(16, 16)
    metrics = compute_metrics(input_estimate, input_reference)

    np.savez(
        os.path.join(vis_dir, "metrics.npz"),
        input_estimate=input_estimate,
        input_reference=input_reference,
        output_nrmse=np.array(metrics["nrmse"]),
        output_ncc=np.array(metrics["ncc"]),
    )
    print("[OK] visualization/metrics.npz")

    print("\nAll xray_tooth_gridrec fixtures generated successfully!")


if __name__ == "__main__":
    main()
