"""
Dual-Energy CT Material Decomposition

Pipeline: load dual-energy sinograms -> Gauss-Newton decomposition ->
FBP reconstruction of material density maps -> evaluate vs ground truth.
"""

import os
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Solver parameters (not in meta_data.json per project convention)
# ---------------------------------------------------------------------------
_GN_PARAMS = {
    "n_iters": 20,        # Gauss-Newton iterations per sinogram view
    "eps": 1e-6,          # initial material line-integral value
    "dE": 1.0,            # energy bin width (keV)
}


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(task_dir, "data")
    output_dir = os.path.join(task_dir, "output")
    ref_dir = os.path.join(task_dir, "evaluation", "reference_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 0: Generate data if it doesn't exist
    # ------------------------------------------------------------------
    if not os.path.exists(os.path.join(data_dir, "raw_data.npz")):
        print("Generating synthetic data...")
        from src.generate_data import generate_synthetic_data, save_task_data
        data = generate_synthetic_data()
        save_task_data(data, task_dir)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    from src.preprocessing import load_raw_data, load_ground_truth, load_metadata
    sinograms, spectra, mus, energies, theta = load_raw_data(data_dir)
    tissue_ref, bone_ref, tissue_sino_ref, bone_sino_ref = load_ground_truth(data_dir)
    meta = load_metadata(data_dir)

    image_size = meta["image_size"]
    print(f"  Sinogram shape: {sinograms.shape}")
    print(f"  Image size: {image_size}")
    print(f"  Number of angles: {len(theta)}")
    print(f"  Energy bins: {len(energies)}")

    # ------------------------------------------------------------------
    # Step 2: Gauss-Newton material decomposition
    # ------------------------------------------------------------------
    print("\nRunning Gauss-Newton material decomposition...")
    from src.solvers import gauss_newton_decompose, reconstruct_material_maps

    t0 = time.time()
    material_sinos = gauss_newton_decompose(
        sinograms, spectra, mus,
        n_iters=_GN_PARAMS["n_iters"],
        dE=_GN_PARAMS["dE"],
        eps=_GN_PARAMS["eps"],
        verbose=True,
    )
    t_decomp = time.time() - t0
    print(f"  Decomposition time: {t_decomp:.1f}s")
    print(f"  Material sinogram shape: {material_sinos.shape}")

    # ------------------------------------------------------------------
    # Step 3: FBP reconstruction of material maps
    # ------------------------------------------------------------------
    print("\nReconstructing material density maps (FBP)...")
    material_maps = reconstruct_material_maps(material_sinos, theta, image_size)
    tissue_est = material_maps[0]
    bone_est = material_maps[1]

    # Clip negative values (physically density >= 0)
    tissue_est = np.clip(tissue_est, 0, None)
    bone_est = np.clip(bone_est, 0, None)

    print(f"  Tissue map range: [{tissue_est.min():.3f}, {tissue_est.max():.3f}]")
    print(f"  Bone map range:   [{bone_est.min():.3f}, {bone_est.max():.3f}]")

    # ------------------------------------------------------------------
    # Step 4: Evaluate
    # ------------------------------------------------------------------
    print("\nComputing metrics...")
    from src.visualization import (compute_metrics, plot_material_maps,
                                   plot_sinograms, plot_spectra_and_mac)

    metrics = compute_metrics(tissue_est, tissue_ref, bone_est, bone_ref)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Step 5: Save outputs
    # ------------------------------------------------------------------
    print("\nSaving outputs...")

    # Save reconstructed maps
    np.savez(
        os.path.join(output_dir, "reconstructed_maps.npz"),
        tissue_map=tissue_est,
        bone_map=bone_est,
        tissue_sinogram=material_sinos[0],
        bone_sinogram=material_sinos[1],
    )

    # Save reference outputs
    os.makedirs(ref_dir, exist_ok=True)
    np.savez(
        os.path.join(ref_dir, "reference_reconstruction.npz"),
        tissue_map=tissue_est[np.newaxis],
        bone_map=bone_est[np.newaxis],
        tissue_sinogram=material_sinos[0][np.newaxis],
        bone_sinogram=material_sinos[1][np.newaxis],
    )

    # Save metrics
    metrics_path = os.path.join(task_dir, "evaluation", "metrics.json")
    metrics_json = {
        "baseline": [
            {
                "method": "Gauss-Newton sinogram decomposition + FBP",
                "tissue_ncc_vs_ref": metrics["tissue_ncc"],
                "tissue_nrmse_vs_ref": metrics["tissue_nrmse"],
                "bone_ncc_vs_ref": metrics["bone_ncc"],
                "bone_nrmse_vs_ref": metrics["bone_nrmse"],
                "ncc_vs_ref": metrics["mean_ncc"],
                "nrmse_vs_ref": metrics["mean_nrmse"],
            }
        ],
        "ncc_boundary": round(0.9 * metrics["mean_ncc"], 4),
        "nrmse_boundary": round(1.1 * metrics["mean_nrmse"], 4),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Plots
    plot_material_maps(tissue_est, bone_est, tissue_ref, bone_ref,
                       save_path=os.path.join(output_dir, "material_maps.png"))
    plot_sinograms(sinograms[0], sinograms[1],
                   save_path=os.path.join(output_dir, "sinograms.png"))
    plot_spectra_and_mac(energies, spectra, mus,
                         save_path=os.path.join(output_dir, "spectra_mac.png"))
    plt.close("all")
    print("  Figures saved to output/")

    print("\nDone.")
    return metrics


if __name__ == "__main__":
    main()
