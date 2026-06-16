"""
Seismic Traveltime Tomography via Adjoint-State Method
=======================================================

Pipeline for 2D isotropic velocity tomography using ray-based
sensitivity kernels and gradient descent.

Steps:
    1. Load and preprocess FPM measurement data
    2. Build the initial (background) velocity model
    3. Run ATT inversion (Eikonal forward + ray back-projection + gradient descent)
    4. Evaluate recovered velocity against ground truth
    5. Save results and visualizations

Usage:
    cd tasks/seismic_traveltime_tomography
    python main.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')

from src.preprocessing import load_data
from src.generate_data import (
    generate_data,
    make_background_velocity,
    DEFAULT_NX, DEFAULT_NZ, DEFAULT_DX_KM, DEFAULT_DZ_KM,
    DEFAULT_V0, DEFAULT_V1,
)
from src.solvers import ATTSolver
from src.visualization import (
    plot_checkerboard_recovery,
    plot_convergence,
    plot_sensitivity_kernel,
    evaluate_reconstruction,
)

# Inversion hyperparameters — not in meta_data.json to avoid leaking solver params
_ATT_PARAMS = {
    'num_iterations': 80,
    'step_size':      0.02,   # max fractional slowness change per iteration
    'step_decay':     0.97,   # per-iteration step size decay factor
    'zeta':           0.5,    # kernel density normalisation exponent
    'epsilon':        1e-4,   # KD normalisation floor
    'step_km':        1.0,    # ray tracing step length (km)
    'smooth_sigma':   4.0,    # Gaussian smoothing sigma (grid cells)
}


def main():
    output_dir = 'output'
    data_dir   = 'data'
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load (or generate) data ─────────────────────────────────────
    raw_path = os.path.join(data_dir, 'raw_data.npz')
    if not os.path.exists(raw_path):
        print("Data not found — generating synthetic dataset...")
        generate_data(data_dir)

    print("Step 1: Loading data...")
    data      = load_data(data_dir)
    T_obs     = data['traveltime_obs']   # (N_src, N_rec)
    sources   = data['sources']           # (N_src, 2) km
    receivers = data['receivers']         # (N_rec, 2) km
    v_true    = data['velocity_true']     # (Nz, Nx) km/s
    meta      = data['meta']

    Nx = meta['Nx'];  Nz = meta['Nz']
    dx = meta['dx_km'];  dz = meta['dz_km']
    print(f"  Grid: {Nx}×{Nz}, spacing {dx} km × {dz} km")
    print(f"  Sources: {len(sources)}, Receivers: {len(receivers)}")

    # ── Step 2: Build initial (background) velocity model ───────────────────
    print("\nStep 2: Building initial model...")
    v_init = make_background_velocity(Nx, Nz, dx, dz,
                                      meta['v0_km_s'], meta['v1_km_s'])
    s_init = (1.0 / v_init).astype(np.float32)
    print(f"  Velocity range: {v_init.min():.2f} – {v_init.max():.2f} km/s")

    # ── Step 3: Run ATT inversion ─────────────────────────────────────────────
    print(f"\nStep 3: Running ATT inversion ({_ATT_PARAMS['num_iterations']} iterations)...")
    solver = ATTSolver(**_ATT_PARAMS)
    results = solver.run(
        slowness_init=s_init,
        dx=dx, dz=dz,
        sources=sources,
        receivers=receivers,
        T_obs=T_obs,
        verbose=True,
    )

    v_inv   = results['velocity']
    history = results['misfit_history']
    kernel  = results['kernel_final']

    print(f"\n  Initial misfit: {history[0]:.4e} s²")
    print(f"  Final misfit:   {history[-1]:.4e} s²")
    print(f"  Reduction:      {history[0] / history[-1]:.1f}×")

    # ── Step 4: Evaluate ─────────────────────────────────────────────────────
    print("\nStep 4: Evaluating reconstruction quality...")
    metrics = evaluate_reconstruction(v_inv, v_true, v_init)
    print(f"  NCC  (perturbation): {metrics['ncc']:.4f}")
    print(f"  NRMSE (perturbation): {metrics['nrmse']:.4f}")
    print(f"  NCC  (full model):   {metrics['ncc_full']:.4f}")
    print(f"  NRMSE (full model):  {metrics['nrmse_full']:.4f}")

    # ── Step 5: Save results ──────────────────────────────────────────────────
    print("\nStep 5: Saving results...")

    np.savez(
        os.path.join(output_dir, 'velocity_reconstructed.npz'),
        velocity=v_inv[np.newaxis],
        velocity_perturbation=(v_inv - v_init)[np.newaxis],
    )

    plot_checkerboard_recovery(
        v_true, v_init, v_inv, dx, dz,
        sources=sources, receivers=receivers,
        metrics=metrics,
        save_path=os.path.join(output_dir, 'velocity_recovery.png'),
    )
    plot_convergence(
        history,
        save_path=os.path.join(output_dir, 'convergence.png'),
    )
    plot_sensitivity_kernel(
        kernel, dx, dz,
        save_path=os.path.join(output_dir, 'sensitivity_kernel.png'),
    )

    results_json = {
        'ncc_perturbation':   metrics['ncc'],
        'nrmse_perturbation': metrics['nrmse'],
        'ncc_full':           metrics['ncc_full'],
        'nrmse_full':         metrics['nrmse_full'],
        'initial_misfit':     history[0],
        'final_misfit':       history[-1],
        'misfit_reduction':   history[0] / history[-1],
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"  Results saved to {output_dir}/")
    print("\nDone!")
    return results_json


if __name__ == '__main__':
    main()
