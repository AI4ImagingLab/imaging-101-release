"""
Generate evaluation fixtures for photoacoustic_tomography task.

Run from the task directory:
    cd photoacoustic_tomography
    python generate_fixtures.py
"""

import os
import sys
import numpy as np

# Ensure src is importable
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from src.generate_data import define_targets, define_detector_array, define_time_vector
from src.physics_model import (
    step_function, pa_signal_single_target,
    generate_ground_truth_image, simulate_pa_signals,
)
from src.solvers import universal_back_projection
from src.visualization import compute_ncc, compute_nrmse

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)


def gen_generate_data_targets():
    """Fixture: generate_data_targets.npz"""
    tar_info, z_target = define_targets()
    path = os.path.join(FIXTURE_DIR, "generate_data_targets.npz")
    np.savez(path, output_tar_info=tar_info, output_z_target=np.float64(z_target))
    print(f"  Saved {path}")


def gen_generate_data_detector():
    """Fixture: generate_data_detector.npz"""
    xd, yd = define_detector_array()
    path = os.path.join(FIXTURE_DIR, "generate_data_detector.npz")
    np.savez(path, output_xd=xd, output_yd=yd)
    print(f"  Saved {path}")


def gen_generate_data_time():
    """Fixture: generate_data_time.npz"""
    t, fs = define_time_vector()
    path = os.path.join(FIXTURE_DIR, "generate_data_time.npz")
    np.savez(path, output_t=t, output_fs=np.float64(fs))
    print(f"  Saved {path}")


def gen_physics_model_step_function():
    """Fixture: physics_model_step_function.npz"""
    input_x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    output_y = step_function(input_x)
    path = os.path.join(FIXTURE_DIR, "physics_model_step_function.npz")
    np.savez(path, input_x=input_x, output_y=output_y)
    print(f"  Saved {path}")


def gen_physics_model_single_target():
    """Fixture: physics_model_single_target.npz

    Use a small detector array (3x3) for speed.
    The tests use params from the fixture, so sizes can be small.
    Target at origin, z=15mm so it's on-axis for the center detector,
    and symmetric for corner detectors.
    """
    tar_info = np.array([0.0, 0.0, 15e-3, 0.75e-3])  # on-axis target
    xd = np.array([-1e-3, 0.0, 1e-3])
    yd = np.array([-1e-3, 0.0, 1e-3])
    t, _ = define_time_vector()
    c = 1484.0

    print("  Computing single-target PA signal (3x3 detector, may take a moment)...")
    signal = pa_signal_single_target(tar_info, xd, yd, t, c)

    path = os.path.join(FIXTURE_DIR, "physics_model_single_target.npz")
    np.savez(path,
             param_tar_info=tar_info,
             param_xd=xd,
             param_yd=yd,
             param_t=t,
             param_c=np.float64(c),
             output_signal=signal)
    print(f"  Saved {path}  signal shape={signal.shape}")


def gen_physics_model_ground_truth():
    """Fixture: physics_model_ground_truth.npz"""
    tar_info, z_target = define_targets()
    # Use a coarse grid for speed
    xd, yd = define_detector_array()
    resolution = 500e-6
    xf = np.arange(xd[0], xd[-1] + resolution, resolution)
    yf = xf.copy()

    gt = generate_ground_truth_image(tar_info, xf, yf)

    path = os.path.join(FIXTURE_DIR, "physics_model_ground_truth.npz")
    np.savez(path,
             param_tar_info=tar_info,
             param_xf=xf,
             param_yf=yf,
             output_gt=gt)
    print(f"  Saved {path}  gt shape={gt.shape}")


def gen_physics_model_simulate():
    """Fixture: physics_model_simulate.npz

    Use 2 targets on a very small 3x3 detector grid with short time vector
    so the test is fast.
    """
    tar_info = np.array([
        [-2e-3, 0.0, 15e-3, 0.75e-3],
        [2e-3, 0.0, 15e-3, 0.75e-3],
    ])
    xd = np.array([-1e-3, 0.0, 1e-3])
    yd = np.array([-1e-3, 0.0, 1e-3])
    # Use a short time vector for speed
    fs = 20e6
    ts = 1.0 / fs
    t = np.arange(0, 20e-6 + ts, ts)

    print("  Computing multi-target PA signal (2 targets, 3x3 detector)...")
    signals = simulate_pa_signals(tar_info, xd, yd, t)

    path = os.path.join(FIXTURE_DIR, "physics_model_simulate.npz")
    np.savez(path,
             input_tar_info=tar_info,
             input_xd=xd,
             input_yd=yd,
             input_t=t,
             output_signals=signals)
    print(f"  Saved {path}  signals shape={signals.shape}")


def gen_solver_ubp():
    """Fixture: solver_ubp.npz

    Use a tiny scenario: 1 off-axis target, 3x3 detector, short time vector.
    The test uses resolution=1e-3 so the image grid is small.
    Target is offset from center to avoid tied-max pixels in normalization.
    """
    tar_info = np.array([[0.5e-3, 0.5e-3, 15e-3, 0.75e-3]])
    xd = np.array([-2e-3, 0.0, 2e-3])
    yd = np.array([-2e-3, 0.0, 2e-3])
    fs = 20e6
    ts = 1.0 / fs
    t = np.arange(0, 20e-6 + ts, ts)
    z_target = 15e-3

    print("  Computing UBP signals for solver fixture (1 target, 3x3)...")
    signals = simulate_pa_signals(tar_info, xd, yd, t)

    print("  Running UBP reconstruction (resolution=1e-3)...")
    recon, xf, yf, zf = universal_back_projection(
        signals, xd, yd, t, z_target, resolution=1e-3)

    path = os.path.join(FIXTURE_DIR, "solver_ubp.npz")
    np.savez(path,
             input_signals=signals,
             param_xd=xd,
             param_yd=yd,
             param_t=t,
             param_z_target=np.float64(z_target),
             output_recon=recon)
    print(f"  Saved {path}  recon shape={recon.shape}")


def gen_visualization_metrics():
    """Fixture: visualization_metrics.npz"""
    rng = np.random.default_rng(42)
    input_a = rng.random((10, 10))
    input_b = rng.random((10, 10))

    output_ncc = compute_ncc(input_a, input_b)
    output_nrmse = compute_nrmse(input_a, input_b)

    path = os.path.join(FIXTURE_DIR, "visualization_metrics.npz")
    np.savez(path,
             input_a=input_a,
             input_b=input_b,
             output_ncc=np.float64(output_ncc),
             output_nrmse=np.float64(output_nrmse))
    print(f"  Saved {path}  ncc={output_ncc:.6f} nrmse={output_nrmse:.6f}")


if __name__ == "__main__":
    print("Generating photoacoustic_tomography fixtures...")
    gen_generate_data_targets()
    gen_generate_data_detector()
    gen_generate_data_time()
    gen_physics_model_step_function()
    gen_physics_model_single_target()
    gen_physics_model_ground_truth()
    gen_physics_model_simulate()
    gen_solver_ubp()
    gen_visualization_metrics()
    print("Done!")
