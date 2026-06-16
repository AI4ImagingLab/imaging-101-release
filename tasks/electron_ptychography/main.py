"""Electron Ptychography: main pipeline entry point.

Reconstructs the phase of gold nanoparticles on amorphous carbon from a
defocused 4D-STEM dataset using DPC, parallax, and ptychography.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import (
    load_data,
    load_metadata,
    calibrate_datacube,
    compute_dp_mean,
    compute_virtual_images,
    compute_bf_mask,
)
from src.solvers import solve_dpc, solve_parallax, solve_ptychography
from src.visualization import (
    compute_metrics,
    plot_phase_comparison,
    plot_reconstruction,
    plot_virtual_images,
    print_metrics_table,
)

# ── Configuration ──────────────────────────────────────────────────────
_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_TASK_DIR, "data")
_OUTPUT_DIR = os.path.join(_TASK_DIR, "output")
_REF_DIR = os.path.join(_TASK_DIR, "evaluation", "reference_outputs")

_DP_MASK_THRESHOLD = 0.8

_PTYCHO_PARAMS = {
    "max_iter": 10,
    "step_size": 0.5,
    "batch_fraction": 4,
}


def main():
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # ── 1. Load data and metadata ──────────────────────────────────────
    print("Loading data...")
    datacube, probe = load_data(_DATA_DIR)
    meta = load_metadata(_DATA_DIR)
    print(f"  Datacube: {datacube.shape}, Probe: {probe.shape}")

    # ── 2. Preprocessing ──────────────────────────────────────────────
    print("Preprocessing...")
    dp_mean = compute_dp_mean(datacube)
    probe_radius, probe_center = calibrate_datacube(
        datacube, probe,
        R_pixel_size=meta["R_pixel_size_A"],
        convergence_semiangle=meta["convergence_semiangle_mrad"],
    )
    print(f"  Probe radius: {probe_radius:.2f} px, center: ({probe_center[0]:.2f}, {probe_center[1]:.2f})")

    bf, df = compute_virtual_images(datacube, probe_center, probe_radius)
    plot_virtual_images(bf, df, save_path=os.path.join(_OUTPUT_DIR, "virtual_images.png"))

    dp_mask = compute_bf_mask(dp_mean, threshold=_DP_MASK_THRESHOLD)

    # ── 3. DPC reconstruction ──────────────────────────────────────────
    print("Running DPC...")
    com_rotation = meta["com_rotation_deg"]
    dpc_phase = solve_dpc(datacube, meta["energy_eV"], dp_mask, com_rotation,
                          R_pixel_size=meta["R_pixel_size_A"])
    print(f"  DPC phase range: [{dpc_phase.min():.4f}, {dpc_phase.max():.4f}]")

    # ── 4. Parallax reconstruction ─────────────────────────────────────
    print("Running parallax...")
    parallax_phase, aberrations = solve_parallax(
        datacube, meta["energy_eV"], com_rotation,
        R_pixel_size=meta["R_pixel_size_A"],
    )
    defocus = -aberrations["C1"] if abs(aberrations["C1"]) > 1 else meta["defocus_A"]
    rotation_rads = aberrations["rotation_Q_to_R_rads"]
    transpose = aberrations["transpose"]
    print(f"  Fitted defocus: {defocus:.1f} A")
    print(f"  Fitted rotation: {np.rad2deg(rotation_rads):.1f} deg")

    # ── 5. Ptychographic reconstruction ────────────────────────────────
    print("Running ptychography...")
    ptycho_phase, ptycho_complex, probe_recon, errors = solve_ptychography(
        datacube, probe, meta["energy_eV"],
        defocus=defocus,
        com_rotation=np.rad2deg(rotation_rads),
        transpose=transpose,
        R_pixel_size=meta["R_pixel_size_A"],
        **_PTYCHO_PARAMS,
    )
    print(f"  Final NMSE: {errors[-1]:.6f}")
    print(f"  Ptycho phase range: [{ptycho_phase.min():.4f}, {ptycho_phase.max():.4f}]")

    # ── 6. Visualization ──────────────────────────────────────────────
    print("Generating plots...")
    plot_phase_comparison(
        dpc_phase, parallax_phase, ptycho_phase,
        save_path=os.path.join(_OUTPUT_DIR, "phase_comparison.png"),
    )
    plot_reconstruction(
        ptycho_phase, probe_recon, errors,
        save_path=os.path.join(_OUTPUT_DIR, "reconstruction.png"),
    )

    # ── 7. Save outputs ───────────────────────────────────────────────
    print("Saving outputs...")
    np.save(os.path.join(_OUTPUT_DIR, "ptycho_phase.npy"), ptycho_phase)
    np.save(os.path.join(_OUTPUT_DIR, "ptycho_complex.npy"), ptycho_complex)
    np.save(os.path.join(_OUTPUT_DIR, "dpc_phase.npy"), dpc_phase)
    np.save(os.path.join(_OUTPUT_DIR, "parallax_phase.npy"), parallax_phase)
    np.save(os.path.join(_OUTPUT_DIR, "probe_recon.npy"), probe_recon)

    # ── 8. Compute and save metrics ────────────────────────────────────
    ref_path = os.path.join(_REF_DIR, "ptycho_phase.npy")
    if os.path.exists(ref_path):
        ref_phase = np.load(ref_path)
        metrics = {}

        # Center-crop both to the smaller common size for comparison
        def _center_crop(a, b):
            """Crop both arrays to the smaller common center region."""
            s = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
            def _crop(x, shape):
                slices = tuple(
                    slice((xs - ss) // 2, (xs - ss) // 2 + ss)
                    for xs, ss in zip(x.shape, shape)
                )
                return x[slices]
            return _crop(a, s), _crop(b, s)

        est_crop, ref_crop = _center_crop(ptycho_phase, ref_phase)
        metrics["ptychography"] = compute_metrics(est_crop, ref_crop)

        print_metrics_table(metrics)

        metrics_out = {
            "ptychography": metrics["ptychography"],
            "final_nmse": errors[-1],
        }
        with open(os.path.join(_OUTPUT_DIR, "metrics.json"), "w") as f:
            json.dump(metrics_out, f, indent=2)
    else:
        print("  No reference outputs found; skipping metric computation.")
        print(f"  Run this pipeline first to generate reference at {_REF_DIR}")

    print("Done.")


if __name__ == "__main__":
    main()
