"""Single-Molecule Light Field Microscopy (SMLFM) reconstruction pipeline.

Usage:
    python main.py

Outputs:
    evaluation/reference_outputs/locs_3d.csv  -- 3D localisations table
    evaluation/reference_outputs/metrics.json -- summary statistics
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import load_localizations, center_localizations
from src.physics_model import build_microscope, build_mla, assign_to_lenses, compute_alpha_model
from src.solvers import fit_aberrations, fit_3d_localizations
from src.visualization import plot_mla_alignment, plot_3d_locs, plot_histograms, plot_occurrences

TASK_DIR = Path(__file__).parent
DATA_DIR = TASK_DIR / "data"
OUTPUT_DIR = TASK_DIR / "evaluation" / "reference_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 0  # (unused; kept for API compatibility)


def main():
    t0 = time.time()

    # --- Load configuration ---
    with open(DATA_DIR / "meta_data.json") as f:
        meta = json.load(f)

    # Fitting parameters (hard-coded defaults; removed from meta_data.json to avoid leaking solver params)
    meta.setdefault("fit_params_aberration", {
        "frame_min": -1,
        "frame_max": -1,
        "disparity_max": 5.0,
        "disparity_step": 0.1,
        "dist_search": 0.5,
        "angle_tolerance": 2.0,
        "threshold": 1.0,
        "min_views": 3,
        "z_calib": None,
    })
    meta.setdefault("aberration_params", {
        "axial_window": 1.0,
        "photon_threshold": 1,
        "min_views": 3,
    })
    meta.setdefault("fit_params_full", {
        "frame_min": -1,
        "frame_max": -1,
        "disparity_max": 8.0,
        "disparity_step": 0.1,
        "dist_search": 0.5,
        "angle_tolerance": 1.0,
        "threshold": 0.3,
        "min_views": 2,
        "z_calib": 1.534,
    })

    # --- Step 1: Preprocessing ---
    print("Loading 2D localisations...")
    locs_2d_csv = load_localizations(DATA_DIR / "raw_data.npz", meta)
    locs_2d_csv = center_localizations(locs_2d_csv)
    n_frames = int(np.unique(locs_2d_csv[:, 0]).shape[0])
    print(f"  {locs_2d_csv.shape[0]:,} localisations in {n_frames:,} frames")

    # --- Step 2: Build physics model ---
    print("Building microscope and MLA...")
    lfm = build_microscope(meta)
    mla = build_mla(meta)
    print(f"  BFP radius: {lfm.bfp_radius:.1f} um, {lfm.bfp_lens_count:.1f} lenses across")
    print(f"  Pixel size in sample: {lfm.pixel_size_sample:.4f} um")
    print(f"  Magnification: {lfm.magnification:.2f}x")

    # --- Step 3: Assign localisations to microlenses ---
    print("Assigning localisations to microlenses...")
    t = time.time()
    lfl = assign_to_lenses(locs_2d_csv, mla, lfm)
    print(f"  Done in {time.time()-t:.1f} s")

    # --- Step 4: Filter localisations ---
    lfl.filter_lenses(mla, lfm)
    if meta.get("filter_rhos") is not None:
        lfl.filter_rhos(tuple(meta["filter_rhos"]))
    if meta.get("filter_spot_sizes") is not None:
        lfl.filter_spot_sizes(tuple(meta["filter_spot_sizes"]))
    if meta.get("filter_photons") is not None:
        lfl.filter_photons(tuple(meta["filter_photons"]))
    print(f"  After filtering: {lfl.filtered_locs_2d.shape[0]:,} localisations")

    # --- Step 5: Compute alpha model ---
    print(f"Computing alpha model ({meta['alpha_model']})...")
    t = time.time()
    compute_alpha_model(lfl, lfm, meta["alpha_model"])
    print(f"  Done in {time.time()-t:.1f} s")

    # --- Step 6: Aberration correction ---
    print("Fitting for aberration correction (first 1000 frames)...")
    t = time.time()
    correction = fit_aberrations(lfl, lfm, meta)
    lfl.correct_xy(correction)
    print(f"  Done in {time.time()-t:.1f} s")
    mean_corr = np.mean(np.sqrt(correction[:, 2]**2 + correction[:, 3]**2))
    print(f"  {correction.shape[0]} views, mean correction: {1000*mean_corr:.1f} nm")

    # --- Step 7: Full 3D fitting ---
    print("Fitting full dataset...")
    t = time.time()

    def progress(frame, min_frame, max_frame):
        pct = 100 * (frame - min_frame) / max(1, max_frame - min_frame)
        print(f"  Frame {frame}/{max_frame} ({pct:.0f}%)", end="\r")

    locs_3d = fit_3d_localizations(lfl, lfm, meta, progress_func=progress)
    print(f"\n  Done in {time.time()-t:.1f} s")
    print(f"  3D localisations: {locs_3d.shape[0]:,}")
    print(f"  2D locs used:     {int(np.sum(locs_3d[:, 5])):,}")

    # --- Step 8: Save outputs ---
    locs_3d_path = OUTPUT_DIR / "locs_3d.csv"
    header = "x_um,y_um,z_um,lateral_err_um,axial_err_um,n_views,photons,frame"
    np.savetxt(locs_3d_path, locs_3d, delimiter=",", header=header, comments="")
    print(f"Saved 3D localisations to {locs_3d_path}")

    lat_err_nm = 1000 * locs_3d[:, 3]
    ax_err_nm  = 1000 * locs_3d[:, 4]
    n_views_col = locs_3d[:, 5]
    keep = ((lat_err_nm < meta.get("show_max_lateral_err", 200.0))
            & (n_views_col > meta.get("show_min_view_count", 3)))

    metrics = {
        "n_locs_3d_total":      int(locs_3d.shape[0]),
        "n_locs_3d_filtered":   int(np.sum(keep)),
        "median_lateral_err_nm": float(np.median(lat_err_nm[keep])),
        "median_axial_err_nm":   float(np.median(ax_err_nm[keep])),
        "mean_lateral_err_nm":   float(np.mean(lat_err_nm[keep])),
        "mean_axial_err_nm":     float(np.mean(ax_err_nm[keep])),
        "median_photons":        float(np.median(locs_3d[keep, 6])),
        "median_n_views":        float(np.median(locs_3d[keep, 5])),
        "z_range_um":            [float(np.min(locs_3d[keep, 2])),
                                   float(np.max(locs_3d[keep, 2]))],
        "total_time_s":          round(time.time() - t0, 1),
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: median lateral={metrics['median_lateral_err_nm']:.0f} nm, "
          f"axial={metrics['median_axial_err_nm']:.0f} nm")

    # --- Step 9: Figures ---
    print("Saving figures...")
    fig = plt.figure(figsize=(6, 6))
    plot_3d_locs(fig, locs_3d,
                 max_lateral_err=meta.get("show_max_lateral_err"),
                 min_views=meta.get("show_min_view_count"))
    fig.savefig(OUTPUT_DIR / "fig_3d_locs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(12, 4))
    plot_histograms(fig, locs_3d,
                    max_lateral_err=meta.get("show_max_lateral_err"),
                    min_views=meta.get("show_min_view_count"))
    fig.savefig(OUTPUT_DIR / "fig_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(6, 5))
    plot_occurrences(fig, locs_3d,
                     max_lateral_err=meta.get("show_max_lateral_err"),
                     min_views=meta.get("show_min_view_count"))
    fig.savefig(OUTPUT_DIR / "fig_occurrences.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Total time: {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
