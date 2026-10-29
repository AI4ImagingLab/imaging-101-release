"""Generate test fixtures for evaluation/fixtures/.

Run once on a machine with CUDA to create pickle fixtures for function-level testing.

Usage:
    cd tasks/usct_FWI
    python generate_fixtures.py
"""

import os
import pickle
import numpy as np
import torch

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)


def save_fixture(name, data):
    path = os.path.join(FIXTURE_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved {name}")


def generate_preprocessing_fixtures():
    """Fixtures for preprocessing module."""
    print("Generating preprocessing fixtures...")

    # --- build_restriction_operator ---
    ix = np.array([10, 30, 50, 60], dtype=np.float32)
    iy = np.array([15, 35, 45, 55], dtype=np.float32)
    nx, ny = 64, 64

    save_fixture("input_build_restriction_operator.pkl", {
        "ix": ix, "iy": iy, "nx": nx, "ny": ny
    })

    from src.preprocessing import build_restriction_operator
    R = build_restriction_operator(ix, iy, nx, ny)
    save_fixture("output_build_restriction_operator.pkl", {
        "R_dense": R.to_dense().cpu().numpy(),
        "shape": tuple(R.shape),
    })

    # --- create_dobs_masks ---
    n_rec = 16
    np.random.seed(42)
    ix_mask = np.linspace(20, 460, n_rec).astype(np.float32)
    iy_mask = np.linspace(20, 460, n_rec).astype(np.float32)
    dobs_small = torch.randn(n_rec, n_rec, dtype=torch.complex64).cuda()
    # Zero out some entries to test mask_misfit logic
    dobs_small[0, 0] = 0
    dobs_small[5, 5] = 0

    save_fixture("input_create_dobs_masks.pkl", {
        "dobs": dobs_small.cpu().numpy(),
        "ix": ix_mask, "iy": iy_mask,
        "dh": 50.0, "mute_dist": 7500.0,
    })

    from src.preprocessing import create_dobs_masks
    dobs_masked, mask_esi, mask_misfit = create_dobs_masks(
        dobs_small, ix_mask, iy_mask, 50.0, 7500.0
    )
    save_fixture("output_create_dobs_masks.pkl", {
        "dobs_masked": dobs_masked.cpu().numpy(),
        "mask_esi": mask_esi.cpu().numpy(),
        "mask_misfit": mask_misfit.cpu().numpy(),
    })


def generate_physics_model_fixtures():
    """Fixtures for physics_model (CBS solver) module."""
    print("Generating physics_model fixtures...")

    from src.physics_model import setup_domain, cbs_solve

    # --- setup_domain ---
    velocity = torch.ones(64, 64, device="cuda") * 1500.0
    config = {"freq": 3.0, "dh": 50, "ppw": 8, "lamb": 1,
              "boundary_widths": 5, "born_max": 100, "energy_threshold": 1e-4}

    save_fixture("config_cbs_solve.pkl", {
        "velocity_shape": (64, 64),
        "velocity_value": 1500.0,
        **config,
    })

    domain = setup_domain(velocity, **config)

    # --- cbs_solve ---
    src_ix, src_iy = 32, 32
    field = cbs_solve(src_ix, src_iy, domain)

    save_fixture("input_cbs_solve.pkl", {
        "src_ix": src_ix, "src_iy": src_iy,
    })
    save_fixture("output_cbs_solve.pkl", {
        "field": field.cpu().numpy(),
        "field_shape": tuple(field.shape),
        "field_abs_max": float(torch.abs(field).max().item()),
    })


def generate_solvers_fixtures():
    """Fixtures for solvers module."""
    print("Generating solvers fixtures...")

    from src.solvers import create_gaussian_kernel

    # --- create_gaussian_kernel ---
    save_fixture("input_gaussian_kernel.pkl", {
        "kernel_size": 9, "sigma": 2.0,
    })
    kernel = create_gaussian_kernel(9, 2.0)
    save_fixture("output_gaussian_kernel.pkl", {
        "kernel": kernel.numpy(),
    })


def generate_parity_fixtures():
    """Save parity reference from original code run."""
    print("Generating parity fixtures...")
    import scipy.io as scio

    ref_path = os.path.join(
        os.path.dirname(TASK_DIR), "..", "..",
        "CBS_FWI_torch", "result", "CBS_FWI_torch", "version_7", "Iter0.3.mat"
    )
    # Also try absolute path
    abs_path = "/home/pisquare/zhijun/usct/CBS_FWI_torch/result/CBS_FWI_torch/version_7/Iter0.3.mat"
    for p in [ref_path, abs_path]:
        if os.path.exists(p):
            mat = scio.loadmat(p)
            vp_orig = np.array(mat["Vp_py"], dtype=np.float32)
            save_fixture("output_parity_single_freq.pkl", {
                "vp_original": vp_orig,
                "freq_mhz": 0.3,
                "description": "Original CBS_FWI_torch version_7 Iter0.3 output",
            })
            return
    print("  WARNING: Original reference not found, skipping parity fixture")


if __name__ == "__main__":
    generate_preprocessing_fixtures()
    generate_physics_model_fixtures()
    generate_solvers_fixtures()
    generate_parity_fixtures()
    print(f"\nAll fixtures saved to {FIXTURE_DIR}")
    print(f"Files: {os.listdir(FIXTURE_DIR)}")
