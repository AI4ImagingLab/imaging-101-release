"""
Raman Cell Phenotyping — end-to-end pipeline.

Hyperspectral unmixing of volumetric Raman spectroscopy data to identify
the biomolecular composition (DNA, protein, lipids, cytoplasm) of THP-1
cells using N-FINDR endmember extraction and FCLS abundance estimation.
"""

import json
import os
import random

import matplotlib
matplotlib.use("Agg")

import numpy as np

from src.preprocessing import load_observation, load_metadata, preprocess_volume
from src.solvers import unmix
from src.physics_model import reconstruction_error
from src.visualization import (
    compute_metrics,
    plot_spectra,
    plot_band_image,
    plot_abundance_maps,
    plot_merged_reconstruction,
)

# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------
_N_ENDMEMBERS = 5
_SELECTED_IMAGE_LAYER = 5
_SELECTED_ENDMEMBER_INDICES = [2, 0, 3, 1]
_ENDMEMBER_LABELS = ["Lipids", "Nucleus", "Cytoplasm", "Background"]
_BAND_WAVENUMBERS = [789, 1008, 1303]
_BAND_COMPONENTS = ["DNA", "Protein", "Lipids"]


def main():
    random.seed(12345)
    np.random.seed(12345)
    os.makedirs("output", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    obs = load_observation("data")
    meta = load_metadata("data")
    volume = obs["spectral_volume"]
    spectral_axis = obs["spectral_axis"]
    print(f"  Volume shape: {volume.shape}, spectral axis: {spectral_axis.shape}")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print("Preprocessing...")
    processed, proc_axis = preprocess_volume(volume, spectral_axis)
    print(f"  Processed shape: {processed.shape}, axis range: "
          f"{proc_axis[0]:.1f}-{proc_axis[-1]:.1f} cm-1")

    # ------------------------------------------------------------------
    # 3. Spectral unmixing
    # ------------------------------------------------------------------
    print(f"Unmixing with N-FINDR ({_N_ENDMEMBERS} endmembers) + FCLS...")
    abundance_maps, endmembers = unmix(processed, n_endmembers=_N_ENDMEMBERS)
    print(f"  {len(endmembers)} endmembers extracted, "
          f"abundance map shape: {abundance_maps[0].shape}")

    # ------------------------------------------------------------------
    # 4. Compute metrics
    # ------------------------------------------------------------------
    print("Computing reconstruction error...")
    flat_obs = processed.reshape(-1, processed.shape[-1])
    endmember_matrix = np.stack(endmembers)
    flat_abund = np.stack([a.ravel() for a in abundance_maps], axis=-1)
    rmse = reconstruction_error(flat_obs, endmember_matrix, flat_abund)
    print(f"  RMSE: {rmse:.6f}")

    # Compare against reference if available
    ref_path = "evaluation/reference_outputs"
    metrics_out = {"reconstruction_rmse": rmse}
    if os.path.exists(f"{ref_path}/abundance_maps.npz"):
        ref = np.load(f"{ref_path}/abundance_maps.npz")
        for k, idx in enumerate(_SELECTED_ENDMEMBER_INDICES):
            label = _ENDMEMBER_LABELS[k]
            ref_key = f"abundance_{label.lower()}"
            if ref_key in ref:
                m = compute_metrics(
                    abundance_maps[idx][..., _SELECTED_IMAGE_LAYER],
                    ref[ref_key],
                )
                metrics_out[f"ncc_{label.lower()}"] = m["ncc"]
                metrics_out[f"nrmse_{label.lower()}"] = m["nrmse"]
                print(f"  {label}: NCC={m['ncc']:.4f}, NRMSE={m['nrmse']:.4f}")

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    print("Saving outputs...")
    np.savez(
        "output/reconstruction.npz",
        **{f"abundance_{i}": a for i, a in enumerate(abundance_maps)},
        **{f"endmember_{i}": e for i, e in enumerate(endmembers)},
        spectral_axis=proc_axis,
    )
    with open("output/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
