"""PnP-MSSN: Plug-and-Play MRI Reconstruction with Multiple Self-Similarity Network.

Pipeline entry point. Runs the full reconstruction pipeline:
1. Load and preprocess MRI data
2. Generate radial sampling mask and k-space measurements
3. Compute IFFT baseline reconstruction
4. Run PnP-PGM with MSSN denoiser
5. Evaluate and visualize results
"""

import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")

from src.preprocessing import prepare_data
from src.physics_model import MRIForwardModel
from src.solvers import MSSNDenoiser, pnp_pgm
from src.visualization import (
    compute_metrics,
    plot_comparison,
    plot_convergence,
    plot_error_maps,
    print_metrics_table,
)


def main():
    np.random.seed(0)

    # Set TF environment for legacy Keras compatibility
    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    # ---- Step 1: Data preparation ----
    print("=" * 60)
    print("Step 1: Loading and preparing data")
    print("=" * 60)

    img, mask, y, metadata = prepare_data("data")
    print(f"Image size: {img.shape}")
    print(f"Sampling lines: {metadata['num_lines']}")
    print(f"Mask coverage: {mask.sum() / mask.size:.1%}")

    # ---- Step 2: IFFT baseline ----
    print("\n" + "=" * 60)
    print("Step 2: IFFT baseline reconstruction")
    print("=" * 60)

    model = MRIForwardModel(mask)
    ifft_recon = model.ifft_recon(y)
    ifft_metrics = compute_metrics(img, ifft_recon)
    print(f"IFFT SNR: {ifft_metrics['snr_db']:.2f} dB")

    # ---- Step 3: PnP-MSSN reconstruction ----
    print("\n" + "=" * 60)
    print("Step 3: PnP-MSSN reconstruction")
    print("=" * 60)

    denoiser = MSSNDenoiser(
        image_shape=img.shape,
        sigma=metadata["sigma"],
        model_checkpoints=metadata["model_checkpoints"],
        patch_size=metadata["patch_size"],
        stride=metadata["stride"],
        state_num=metadata["state_num"],
    )

    save_dir = os.path.join("output", "iterations")
    recon, history = pnp_pgm(
        forward_model=model,
        denoiser=denoiser,
        y=y,
        num_iter=metadata["num_iter"],
        step=metadata["step_size"],
        xtrue=img,
        verbose=True,
        save_dir=save_dir,
    )

    # ---- Step 4: Evaluation ----
    print("\n" + "=" * 60)
    print("Step 4: Evaluation")
    print("=" * 60)

    pnp_metrics = compute_metrics(img, recon)
    all_metrics = {"IFFT": ifft_metrics, "PnP-MSSN": pnp_metrics}
    print_metrics_table(all_metrics)

    # ---- Step 5: Save outputs ----
    print("\n" + "=" * 60)
    print("Step 5: Saving outputs")
    print("=" * 60)

    os.makedirs("output", exist_ok=True)
    np.save("output/reconstruction.npy", recon)
    np.save("output/snr_history.npy", np.array(history["snr"]))
    np.save("output/dist_history.npy", np.array(history["dist"]))

    metrics_out = {
        "IFFT": {k: float(v) for k, v in ifft_metrics.items()},
        "PnP-MSSN": {k: float(v) for k, v in pnp_metrics.items()},
    }
    with open("output/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"Reconstruction saved to output/reconstruction.npy")
    print(f"Metrics saved to output/metrics.json")

    # ---- Step 6: Visualization ----
    print("\n" + "=" * 60)
    print("Step 6: Visualization")
    print("=" * 60)

    plot_comparison(img, ifft_recon, recon, all_metrics,
                    save_path="output/comparison.png")
    plot_convergence(history, save_path="output/convergence.png")
    plot_error_maps(img, ifft_recon, recon, mask,
                    save_path="output/error_maps.png")
    print("Figures saved to output/")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
