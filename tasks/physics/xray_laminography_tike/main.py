"""
X-ray Laminography/Tomography Reconstruction
=============================================

Reconstructs a 3D complex-valued volume from simulated laminographic projections
of a nanoparticle assembly phantom using tike's conjugate gradient solver.

Usage:
    cd tasks/xray_laminography_tike
    python main.py
"""

import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASK_DIR = os.path.dirname(os.path.abspath(__file__))

# Solver parameters
N_ROUNDS = 5
N_ITER_PER_ROUND = 4


def main():
    """Run the full reconstruction pipeline."""
    import sys
    sys.path.insert(0, TASK_DIR)

    from src.preprocessing import load_raw_data, load_ground_truth, load_metadata
    from src.preprocessing import create_initial_guess
    from src.solvers import reconstruct
    from src.visualization import compute_metrics, plot_complex_slice, plot_volume_slices

    # --- Load data ---
    print("Loading data...")
    raw = load_raw_data(os.path.join(TASK_DIR, 'data', 'raw_data.npz'))
    gt = load_ground_truth(os.path.join(TASK_DIR, 'data', 'ground_truth.npz'))
    meta = load_metadata(os.path.join(TASK_DIR, 'data', 'meta_data.json'))

    projections = raw['projections']  # (1, 128, 128, 128)
    theta = raw['theta']              # (1, 128)
    gt_volume = gt['volume']          # (1, 128, 128, 128)

    tilt = float(meta['tilt_rad'])
    volume_shape = tuple(meta['volume_shape'])

    # Remove batch dimension for tike
    data = projections[0]   # (128, 128, 128)
    theta_1d = theta[0]     # (128,)
    gt_vol = gt_volume[0]   # (128, 128, 128)

    # --- Reconstruct ---
    print(f"Running reconstruction: {N_ROUNDS} rounds x {N_ITER_PER_ROUND} iterations")
    result = reconstruct(
        data=data,
        theta=theta_1d,
        tilt=tilt,
        volume_shape=volume_shape,
        n_rounds=N_ROUNDS,
        n_iter_per_round=N_ITER_PER_ROUND,
    )

    recon = result['obj']       # (128, 128, 128) complex64
    costs = result['costs']     # list of costs per round

    print(f"Reconstruction complete.")
    print(f"  Cost: {costs[0]:.4f} -> {costs[-1]:.6f}")

    # --- Compute metrics vs ground truth ---
    print("Computing metrics...")
    metrics = compute_metrics(recon, gt_vol)
    print(f"  NCC:   {metrics['ncc']:.4f}")
    print(f"  NRMSE: {metrics['nrmse']:.4f}")

    # --- Save outputs ---
    output_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'reconstructed_volume.npy'), recon)
    np.save(os.path.join(output_dir, 'ground_truth_volume.npy'), gt_vol)
    np.save(os.path.join(output_dir, 'costs.npy'), np.array(costs))

    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Also update top-level evaluation/metrics.json
    eval_metrics = {
        "baseline": [
            {
                "method": f"cgrad ({N_ROUNDS} rounds x {N_ITER_PER_ROUND} iter)",
                "ncc_vs_ref": round(metrics['ncc'], 4),
                "nrmse_vs_ref": round(metrics['nrmse'], 4),
            }
        ],
        "ncc_boundary": round(0.9 * metrics['ncc'], 4),
        "nrmse_boundary": round(1.1 * metrics['nrmse'], 4),
    }
    eval_metrics_path = os.path.join(TASK_DIR, 'evaluation', 'metrics.json')
    with open(eval_metrics_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Saved evaluation metrics to {eval_metrics_path}")

    # --- Visualization ---
    print("Generating visualizations...")
    mid = volume_shape[0] // 2
    slice_indices = [mid - 20, mid, mid + 20]

    # Plot reconstruction slices
    fig = plot_volume_slices(recon, slice_indices)
    fig.suptitle('Reconstruction (real / imaginary)', fontsize=14)
    fig.savefig(os.path.join(output_dir, 'reconstruction_slices.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot ground truth slices
    fig_gt = plot_volume_slices(gt_vol, slice_indices)
    fig_gt.suptitle('Ground Truth (real / imaginary)', fontsize=14)
    fig_gt.savefig(os.path.join(output_dir, 'ground_truth_slices.png'),
                   dpi=150, bbox_inches='tight')
    plt.close(fig_gt)

    # Plot cost convergence
    fig_cost, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(1, len(costs) + 1), costs, 'o-')
    ax.set_xlabel('Round')
    ax.set_ylabel('Cost')
    ax.set_title('Reconstruction Cost Convergence')
    ax.grid(True, alpha=0.3)
    fig_cost.savefig(os.path.join(output_dir, 'cost_convergence.png'),
                     dpi=150, bbox_inches='tight')
    plt.close(fig_cost)

    print("Done.")


if __name__ == '__main__':
    main()
