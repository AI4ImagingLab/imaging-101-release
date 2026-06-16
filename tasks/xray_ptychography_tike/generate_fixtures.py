"""Generate small test fixtures for xray_ptychography_tike.

Run with:
    cd tasks/xray_ptychography_tike
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

    # --- shift_scan_positions ---
    scan_input = np.array([[-10.0, 5.0], [20.0, -3.0], [0.0, 10.0]],
                          dtype=np.float32)
    offset = 20.0
    from src.preprocessing import shift_scan_positions
    scan_output = shift_scan_positions(scan_input, offset=offset)
    np.savez(
        os.path.join(out_dir, "shift_scan_positions.npz"),
        input_scan=scan_input,
        param_offset=np.float32(offset),
        output_scan=scan_output,
    )
    print(f"  shift_scan_positions: input {scan_input.shape} -> output {scan_output.shape}")

    # --- initialize_psi ---
    scan_for_psi = np.array([[20.0, 20.0], [50.0, 60.0]], dtype=np.float32)
    probe_shape = (1, 1, 1, 16, 16)
    n_slices = 1
    buffer = 2
    fill_value = 0.5 + 0j
    from src.preprocessing import initialize_psi
    psi_output = initialize_psi(scan_for_psi, probe_shape,
                                n_slices=n_slices, buffer=buffer,
                                fill_value=fill_value)
    np.savez(
        os.path.join(out_dir, "initialize_psi.npz"),
        input_scan=scan_for_psi,
        param_probe_shape=np.array(probe_shape),
        param_n_slices=np.int32(n_slices),
        param_buffer=np.int32(buffer),
        param_fill_value=np.complex64(fill_value),
        output_psi=psi_output,
    )
    print(f"  initialize_psi: output {psi_output.shape}, dtype={psi_output.dtype}")

    # --- load_raw_data (shape/dtype fixture from real data) ---
    from src.preprocessing import load_raw_data
    raw = load_raw_data(DATA_DIR)
    np.savez(
        os.path.join(out_dir, "load_raw_data.npz"),
        output_diffraction_patterns_shape=np.array(raw["diffraction_patterns"].shape),
        output_diffraction_patterns_dtype=np.array(str(raw["diffraction_patterns"].dtype)),
        output_scan_positions_shape=np.array(raw["scan_positions"].shape),
        output_scan_positions_dtype=np.array(str(raw["scan_positions"].dtype)),
        output_probe_guess_shape=np.array(raw["probe_guess"].shape),
        output_probe_guess_dtype=np.array(str(raw["probe_guess"].dtype)),
    )
    print(f"  load_raw_data: shapes recorded")

    # --- load_metadata ---
    from src.preprocessing import load_metadata
    meta = load_metadata(DATA_DIR)
    # Save as JSON string in npz
    np.savez(
        os.path.join(out_dir, "load_metadata.npz"),
        output_n_positions=np.int32(meta["n_positions"]),
        output_probe_shape=np.array(meta["probe_shape"]),
        output_diffraction_shape=np.array(meta["diffraction_shape"]),
    )
    print(f"  load_metadata: keys recorded")


def generate_physics_model_fixtures():
    """Generate fixtures for src/physics_model.py functions."""
    out_dir = os.path.join(FIXTURES_DIR, "physics_model")
    os.makedirs(out_dir, exist_ok=True)

    # --- validate_inputs (valid case) ---
    N, W, H, S, D = 4, 8, 8, 1, 1
    data_valid = np.random.rand(N, W, H).astype(np.float32)
    scan_valid = np.random.rand(N, 2).astype(np.float32) * 10 + 20
    probe_valid = (np.random.rand(1, 1, S, W, H) + 1j * np.random.rand(1, 1, S, W, H)).astype(np.complex64)
    psi_valid = np.full((D, 50, 50), 0.5 + 0j, dtype=np.complex64)
    np.savez(
        os.path.join(out_dir, "validate_inputs_valid.npz"),
        input_data=data_valid,
        input_scan=scan_valid,
        input_probe=probe_valid,
        input_psi=psi_valid,
    )
    print(f"  validate_inputs (valid): data {data_valid.shape}, scan {scan_valid.shape}")

    # --- validate_inputs (invalid cases - shapes/dtypes for testing) ---
    # wrong data ndim
    np.savez(
        os.path.join(out_dir, "validate_inputs_invalid_data_ndim.npz"),
        input_data=data_valid[0],  # 2D instead of 3D
        input_scan=scan_valid,
        input_probe=probe_valid,
        input_psi=psi_valid,
    )

    # wrong scan shape
    np.savez(
        os.path.join(out_dir, "validate_inputs_invalid_scan_shape.npz"),
        input_data=data_valid,
        input_scan=scan_valid[:, :1],  # (N, 1) instead of (N, 2)
        input_probe=probe_valid,
        input_psi=psi_valid,
    )

    # mismatched N
    np.savez(
        os.path.join(out_dir, "validate_inputs_invalid_n_mismatch.npz"),
        input_data=data_valid[:2],  # N=2 vs N=4
        input_scan=scan_valid,
        input_probe=probe_valid,
        input_psi=psi_valid,
    )
    print(f"  validate_inputs (invalid cases): 3 fixtures")

    # --- simulate_diffraction / create_ptycho_parameters: GPU-dependent ---
    # Save expected shapes/dtypes for GPU tests
    N_sim, W_sim, H_sim = 8, 16, 16
    np.savez(
        os.path.join(out_dir, "simulate_diffraction_shapes.npz"),
        param_N=np.int32(N_sim),
        param_W=np.int32(W_sim),
        param_H=np.int32(H_sim),
        output_data_shape=np.array([N_sim, W_sim, H_sim]),
        output_data_dtype=np.array("float32"),
    )
    np.savez(
        os.path.join(out_dir, "create_ptycho_parameters_config.npz"),
        param_num_iter=np.int32(4),
        param_num_batch=np.int32(2),
        config_algorithm_name=np.array("lstsq_grad"),
    )
    print(f"  simulate_diffraction/create_ptycho_parameters: shape/config fixtures")


def generate_solvers_fixtures():
    """Generate fixtures for src/solvers.py functions."""
    out_dir = os.path.join(FIXTURES_DIR, "solvers")
    os.makedirs(out_dir, exist_ok=True)

    # Solver is GPU-dependent; save expected shapes/dtypes
    np.savez(
        os.path.join(out_dir, "reconstruct_shapes.npz"),
        config_num_iter=np.int32(2),
        config_num_batch=np.int32(2),
        output_psi_ndim=np.int32(3),
        output_probe_ndim=np.int32(5),
        output_probe_batch_dims=np.array([1, 1]),
    )
    print(f"  reconstruct: shape expectations saved")


def generate_visualization_fixtures():
    """Generate fixtures for src/visualization.py functions."""
    out_dir = os.path.join(FIXTURES_DIR, "visualization")
    os.makedirs(out_dir, exist_ok=True)

    from src.visualization import compute_ncc, compute_nrmse, compute_metrics

    # --- compute_ncc ---
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ncc_identical = compute_ncc(a, b)

    c = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    ncc_scaled = compute_ncc(a, c)

    d = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float64)
    ncc_negated = compute_ncc(a, d)

    np.savez(
        os.path.join(out_dir, "compute_ncc.npz"),
        input_a_identical=a,
        input_b_identical=b,
        output_ncc_identical=np.float64(ncc_identical),
        input_a_scaled=a,
        input_c_scaled=c,
        output_ncc_scaled=np.float64(ncc_scaled),
        input_a_negated=a,
        input_d_negated=d,
        output_ncc_negated=np.float64(ncc_negated),
    )
    print(f"  compute_ncc: identical={ncc_identical:.4f}, scaled={ncc_scaled:.4f}, negated={ncc_negated:.4f}")

    # --- compute_nrmse ---
    e = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    f = np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float64)
    nrmse_close = compute_nrmse(f, e)

    nrmse_identical = compute_nrmse(e, e)

    np.savez(
        os.path.join(out_dir, "compute_nrmse.npz"),
        input_estimate_close=f,
        input_reference_close=e,
        output_nrmse_close=np.float64(nrmse_close),
        input_estimate_identical=e,
        input_reference_identical=e,
        output_nrmse_identical=np.float64(nrmse_identical),
    )
    print(f"  compute_nrmse: close={nrmse_close:.6f}, identical={nrmse_identical:.6f}")

    # --- compute_metrics ---
    ref = np.random.RandomState(42).rand(8, 8).astype(np.float32)
    est = ref + np.random.RandomState(43).randn(8, 8).astype(np.float32) * 0.01
    metrics = compute_metrics(est, ref)
    np.savez(
        os.path.join(out_dir, "compute_metrics.npz"),
        input_estimate=est,
        input_reference=ref,
        output_ncc=np.float64(metrics["ncc"]),
        output_nrmse=np.float64(metrics["nrmse"]),
    )
    print(f"  compute_metrics: ncc={metrics['ncc']:.4f}, nrmse={metrics['nrmse']:.6f}")


if __name__ == "__main__":
    np.random.seed(12345)
    print("Generating ptychography test fixtures...")
    print("\n[preprocessing]")
    generate_preprocessing_fixtures()
    print("\n[physics_model]")
    generate_physics_model_fixtures()
    print("\n[solvers]")
    generate_solvers_fixtures()
    print("\n[visualization]")
    generate_visualization_fixtures()
    print("\nDone. Fixtures saved to:", FIXTURES_DIR)
