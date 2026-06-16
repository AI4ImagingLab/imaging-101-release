#!/usr/bin/env python
"""Generate test fixtures for diffusion_mri_dti.

The tests for this task generate their own data inline (they don't load from
evaluation/fixtures/), but we create basic fixtures capturing function outputs
for reference and potential future use.
"""

import os
import sys
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

FIXTURES_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

from src.physics_model import (
    tensor_from_elements,
    elements_from_tensor,
    tensor_from_eig,
    build_design_matrix,
    stejskal_tanner_signal,
    add_rician_noise,
    compute_fa,
    compute_md,
)
from src.generate_data import generate_gradient_table
from src.solvers import fit_dti_ols, fit_dti_wls, tensor_eig_decomposition
from src.visualization import compute_ncc, compute_nrmse


def main():
    print("Generating fixtures for diffusion_mri_dti ...")

    # --- physics_model fixtures ---
    # Tensor construction roundtrip
    D = np.array([[1.0, 0.2, 0.1],
                   [0.2, 0.8, 0.15],
                   [0.1, 0.15, 0.6]])
    elems = elements_from_tensor(D)
    D_reconstructed = tensor_from_elements(*elems)
    np.savez(os.path.join(FIXTURES_DIR, "tensor_roundtrip.npz"),
             input_tensor=D,
             output_elements=elems,
             output_tensor=D_reconstructed)
    print("  [OK] tensor_roundtrip.npz")

    # Design matrix
    bvals = np.array([0, 1000, 1000, 1000], dtype=np.float64)
    bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    B = build_design_matrix(bvals, bvecs)
    np.savez(os.path.join(FIXTURES_DIR, "design_matrix.npz"),
             input_bvals=bvals, input_bvecs=bvecs,
             output=B)
    print("  [OK] design_matrix.npz")

    # Stejskal-Tanner signal
    D_iso = np.diag([1e-3, 1e-3, 1e-3])
    S0 = 1.0
    bvals_st = np.array([0, 500, 1000, 2000], dtype=np.float64)
    bvecs_st = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float64)
    signal = stejskal_tanner_signal(S0, D_iso, bvals_st, bvecs_st)
    np.savez(os.path.join(FIXTURES_DIR, "stejskal_tanner.npz"),
             input_D=D_iso, input_S0=S0,
             input_bvals=bvals_st, input_bvecs=bvecs_st,
             output_signal=signal)
    print("  [OK] stejskal_tanner.npz")

    # FA and MD
    evals = np.array([1.5e-3, 0.5e-3, 0.3e-3])
    fa = compute_fa(evals)
    md = compute_md(evals)
    np.savez(os.path.join(FIXTURES_DIR, "scalar_maps.npz"),
             input_evals=evals,
             output_fa=fa, output_md=md)
    print("  [OK] scalar_maps.npz")

    # --- solvers fixtures ---
    bvals_fit, bvecs_fit = generate_gradient_table(n_directions=30, b_value=1000.0)
    true_elems = [1.2e-3, 0.0, 0.0, 0.4e-3, 0.0, 0.4e-3]
    S0_val = 1.0
    D_true = tensor_from_elements(*true_elems)
    signal_fit = stejskal_tanner_signal(S0_val, D_true, bvals_fit, bvecs_fit)
    dwi = np.zeros((3, 3, len(bvals_fit)), dtype=np.float64)
    dwi[1, 1, :] = signal_fit
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    tensor_ols, S0_ols = fit_dti_ols(dwi, bvals_fit, bvecs_fit, mask=mask)
    tensor_wls, S0_wls = fit_dti_wls(dwi, bvals_fit, bvecs_fit, mask=mask)
    evals_out, evecs_out, fa_map, md_map = tensor_eig_decomposition(tensor_ols, mask=mask)

    np.savez(os.path.join(FIXTURES_DIR, "ols_fit.npz"),
             input_bvals=bvals_fit, input_bvecs=bvecs_fit,
             input_dwi=dwi, input_mask=mask,
             output_tensor=tensor_ols, output_S0=S0_ols)
    print("  [OK] ols_fit.npz")

    np.savez(os.path.join(FIXTURES_DIR, "wls_fit.npz"),
             input_bvals=bvals_fit, input_bvecs=bvecs_fit,
             input_dwi=dwi, input_mask=mask,
             output_tensor=tensor_wls, output_S0=S0_wls)
    print("  [OK] wls_fit.npz")

    np.savez(os.path.join(FIXTURES_DIR, "eig_decomposition.npz"),
             input_tensor_elems=tensor_ols, input_mask=mask,
             output_evals=evals_out, output_evecs=evecs_out,
             output_fa_map=fa_map, output_md_map=md_map)
    print("  [OK] eig_decomposition.npz")

    # --- visualization fixtures ---
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.5, 2.8])
    ncc = compute_ncc(x, y)
    nrmse = compute_nrmse(x, y)
    np.savez(os.path.join(FIXTURES_DIR, "metrics.npz"),
             input_x=x, input_y=y,
             output_ncc=ncc, output_nrmse=nrmse)
    print("  [OK] metrics.npz")

    print("Done! All fixtures saved to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
