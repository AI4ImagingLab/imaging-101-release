"""Generate small test fixtures for xray_laminography_tike.

Run with:
    cd tasks/xray_laminography_tike
    PYTHONPATH=/projects/bgkc/schen47/tike/src:. python3.9 generate_fixtures.py
"""

import os
import sys
import json
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

FIXTURES_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")
DATA_DIR = os.path.join(TASK_DIR, "data")
REF_DIR = os.path.join(TASK_DIR, "evaluation", "reference_outputs")


def generate_preprocessing_fixtures():
    """Generate fixtures for src/preprocessing.py functions."""
    out_dir = os.path.join(FIXTURES_DIR, "preprocessing")
    os.makedirs(out_dir, exist_ok=True)

    from src.preprocessing import load_raw_data, load_ground_truth, load_metadata, create_initial_guess

    # --- load_raw_data (shape/dtype from real data) ---
    raw_path = os.path.join(DATA_DIR, "raw_data.npz")
    raw = load_raw_data(raw_path)
    np.savez(
        os.path.join(out_dir, "load_raw_data.npz"),
        output_projections_shape=np.array(raw["projections"].shape),
        output_projections_dtype=np.array(str(raw["projections"].dtype)),
        output_theta_shape=np.array(raw["theta"].shape),
        output_theta_dtype=np.array(str(raw["theta"].dtype)),
    )
    print(f"  load_raw_data: projections {raw['projections'].shape}, theta {raw['theta'].shape}")

    # --- load_ground_truth (shape/dtype) ---
    gt_path = os.path.join(DATA_DIR, "ground_truth.npz")
    gt = load_ground_truth(gt_path)
    np.savez(
        os.path.join(out_dir, "load_ground_truth.npz"),
        output_volume_shape=np.array(gt["volume"].shape),
        output_volume_dtype=np.array(str(gt["volume"].dtype)),
    )
    print(f"  load_ground_truth: volume {gt['volume'].shape}")

    # --- load_metadata ---
    meta_path = os.path.join(DATA_DIR, "meta_data.json")
    meta = load_metadata(meta_path)
    np.savez(
        os.path.join(out_dir, "load_metadata.npz"),
        output_volume_shape=np.array(meta["volume_shape"]),
        output_n_angles=np.int32(meta["n_angles"]),
        output_tilt_rad=np.float32(meta["tilt_rad"]),
    )
    print(f"  load_metadata: n_angles={meta['n_angles']}, tilt_rad={meta['tilt_rad']:.4f}")

    # --- create_initial_guess ---
    shape_small = (4, 8, 8)
    guess = create_initial_guess(shape_small)
    np.savez(
        os.path.join(out_dir, "create_initial_guess.npz"),
        param_volume_shape=np.array(shape_small),
        output_guess=guess,
    )
    print(f"  create_initial_guess: shape={guess.shape}, dtype={guess.dtype}")

    # Test with custom dtype
    guess_128 = create_initial_guess(shape_small, dtype=np.complex128)
    np.savez(
        os.path.join(out_dir, "create_initial_guess_complex128.npz"),
        param_volume_shape=np.array(shape_small),
        param_dtype=np.array("complex128"),
        output_guess=guess_128,
    )
    print(f"  create_initial_guess (complex128): dtype={guess_128.dtype}")

    # --- load_raw_data with synthetic data ---
    import tempfile
    proj_syn = np.zeros((1, 4, 8, 8), dtype=np.complex64)
    theta_syn = np.linspace(0, np.pi, 4, endpoint=False, dtype=np.float32)[np.newaxis]
    tmp_path = os.path.join(out_dir, "synthetic_raw_data.npz")
    np.savez(tmp_path, projections=proj_syn, theta=theta_syn)
    loaded_syn = load_raw_data(tmp_path)
    np.savez(
        os.path.join(out_dir, "load_raw_data_synthetic.npz"),
        input_projections=proj_syn,
        input_theta=theta_syn,
        output_projections_shape=np.array(loaded_syn["projections"].shape),
        output_theta_shape=np.array(loaded_syn["theta"].shape),
    )
    # Clean up
    os.remove(tmp_path)
    print(f"  load_raw_data (synthetic): verified round-trip")


def generate_physics_model_fixtures():
    """Generate fixtures for src/physics_model.py (GPU-dependent)."""
    out_dir = os.path.join(FIXTURES_DIR, "physics_model")
    os.makedirs(out_dir, exist_ok=True)

    # forward_project is GPU-dependent; save expected shapes/dtypes
    n = 8
    n_angles = 4
    np.savez(
        os.path.join(out_dir, "forward_project_shapes.npz"),
        param_n=np.int32(n),
        param_n_angles=np.int32(n_angles),
        input_obj_shape=np.array([n, n, n]),
        input_theta_shape=np.array([n_angles]),
        output_shape=np.array([n_angles, n, n]),
        output_dtype=np.array("complex64"),
    )
    print(f"  forward_project: expected output ({n_angles}, {n}, {n}) complex64")

    # Assertion test inputs (bad ndim)
    np.savez(
        os.path.join(out_dir, "forward_project_invalid_obj.npz"),
        input_obj=np.zeros((1, 8, 8, 8), dtype=np.complex64),  # 4D, bad
        input_theta=np.array([0.0], dtype=np.float32),
        param_tilt=np.float32(np.pi / 2),
    )
    np.savez(
        os.path.join(out_dir, "forward_project_invalid_theta.npz"),
        input_obj=np.zeros((8, 8, 8), dtype=np.complex64),
        input_theta=np.array([[0.0]], dtype=np.float32),  # 2D, bad
        param_tilt=np.float32(np.pi / 2),
    )
    print(f"  forward_project: invalid input fixtures saved")


def generate_solvers_fixtures():
    """Generate fixtures for src/solvers.py (GPU-dependent)."""
    out_dir = os.path.join(FIXTURES_DIR, "solvers")
    os.makedirs(out_dir, exist_ok=True)

    n = 8
    n_angles = 4
    n_rounds = 2
    n_iter_per_round = 2

    np.savez(
        os.path.join(out_dir, "reconstruct_shapes.npz"),
        param_n=np.int32(n),
        param_n_angles=np.int32(n_angles),
        config_n_rounds=np.int32(n_rounds),
        config_n_iter_per_round=np.int32(n_iter_per_round),
        output_obj_shape=np.array([n, n, n]),
        output_obj_dtype=np.array("complex64"),
        output_costs_length=np.int32(n_rounds),
    )
    print(f"  reconstruct: shape/config expectations saved")

    # Invalid data ndim fixture
    np.savez(
        os.path.join(out_dir, "reconstruct_invalid_data.npz"),
        input_data=np.zeros((1, 4, 8, 8), dtype=np.complex64),  # 4D, bad
        input_theta=np.array([0.0], dtype=np.float32),
        param_tilt=np.float32(np.pi / 2),
        param_volume_shape=np.array([8, 8, 8]),
    )
    print(f"  reconstruct: invalid data fixture saved")


def generate_visualization_fixtures():
    """Generate fixtures for src/visualization.py functions."""
    out_dir = os.path.join(FIXTURES_DIR, "visualization")
    os.makedirs(out_dir, exist_ok=True)

    from src.visualization import compute_metrics

    # --- compute_metrics with complex inputs ---
    rng = np.random.RandomState(42)
    ref_complex = (rng.rand(4, 4, 4) + 1j * rng.rand(4, 4, 4)).astype(np.complex64)
    est_complex = ref_complex + (rng.randn(4, 4, 4) + 1j * rng.randn(4, 4, 4)).astype(np.complex64) * 0.01

    metrics = compute_metrics(est_complex, ref_complex)
    np.savez(
        os.path.join(out_dir, "compute_metrics.npz"),
        input_estimate=est_complex,
        input_reference=ref_complex,
        output_ncc=np.float64(metrics["ncc"]),
        output_nrmse=np.float64(metrics["nrmse"]),
    )
    print(f"  compute_metrics: ncc={metrics['ncc']:.4f}, nrmse={metrics['nrmse']:.6f}")

    # --- compute_metrics with identical arrays ---
    metrics_identical = compute_metrics(ref_complex, ref_complex)
    np.savez(
        os.path.join(out_dir, "compute_metrics_identical.npz"),
        input_estimate=ref_complex,
        input_reference=ref_complex,
        output_ncc=np.float64(metrics_identical["ncc"]),
        output_nrmse=np.float64(metrics_identical["nrmse"]),
    )
    print(f"  compute_metrics (identical): ncc={metrics_identical['ncc']:.4f}, nrmse={metrics_identical['nrmse']:.6f}")

    # --- compute_metrics with zero array ---
    zero_arr = np.zeros((4, 4, 4), dtype=np.complex64)
    metrics_zero = compute_metrics(zero_arr, zero_arr)
    np.savez(
        os.path.join(out_dir, "compute_metrics_zero.npz"),
        input_estimate=zero_arr,
        input_reference=zero_arr,
        output_ncc=np.float64(metrics_zero["ncc"]),
        output_nrmse=np.float64(metrics_zero["nrmse"]),
    )
    print(f"  compute_metrics (zero): ncc={metrics_zero['ncc']:.4f}, nrmse={metrics_zero['nrmse']}")


if __name__ == "__main__":
    np.random.seed(12345)
    print("Generating laminography test fixtures...")
    print("\n[preprocessing]")
    generate_preprocessing_fixtures()
    print("\n[physics_model]")
    generate_physics_model_fixtures()
    print("\n[solvers]")
    generate_solvers_fixtures()
    print("\n[visualization]")
    generate_visualization_fixtures()
    print("\nDone. Fixtures saved to:", FIXTURES_DIR)
