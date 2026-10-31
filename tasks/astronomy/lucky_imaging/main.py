"""
Lucky Imaging: Planetary Surface Reconstruction
================================================

Reconstruct a high-resolution image from short-exposure video frames
degraded by atmospheric turbulence, using frame selection, alignment,
and local adaptive stacking.

Usage:
    cd tasks/lucky_imaging
    python main.py
"""

import os
import sys
import json
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)


def main():
    from src.preprocessing import load_frames, prepare_all_frames
    from src.solvers import (
        rank_frames, find_alignment_rect, align_frames_global,
        create_ap_grid, rank_frames_local, compute_local_shifts,
        stack_and_blend, unsharp_mask,
    )
    from src.visualization import compute_metrics

    data_dir = os.path.join(TASK_DIR, "data")
    output_dir = os.path.join(TASK_DIR, "output")
    ref_dir = os.path.join(TASK_DIR, "evaluation", "reference_outputs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    # Load metadata for image properties
    with open(os.path.join(data_dir, "meta_data.json"), "r") as f:
        meta = json.load(f)
    _PROCESSING_PARAMS = {
        "gauss_width": 7,
        "ranking_method": "Laplace",
        "ranking_stride": 2,
        "normalize_brightness": True,
        "alignment_method": "MultiLevelCorrelation",
        "alignment_search_width": 34,
        "average_frame_percent": 5,
        "ap_half_box_width": 24,
        "ap_search_width": 14,
        "ap_structure_threshold": 0.04,
        "ap_brightness_threshold": 10,
        "ap_frame_percent": 10,
        "drizzle_factor": 1,
        "usm_sigma": 2.0,
        "usm_alpha": 1.5,
    }
    params = _PROCESSING_PARAMS

    # ── [1/8] Load data ──────────────────────────────────────────────────
    print("[1/8] Loading video frames ...")
    t0 = time.time()
    frames, _ = load_frames(data_dir)
    print(f"       Loaded {frames.shape[0]} frames, "
          f"shape {frames.shape[1:]} in {time.time()-t0:.1f}s")

    # ── [2/8] Prepare frames ─────────────────────────────────────────────
    print("[2/8] Preparing frames (mono, blur, Laplacian) ...")
    t0 = time.time()
    frames_data = prepare_all_frames(
        frames,
        gauss_width=params["gauss_width"],
        stride=params["ranking_stride"],
    )
    print(f"       Done in {time.time()-t0:.1f}s")

    # ── [3/8] Rank frames ────────────────────────────────────────────────
    print("[3/8] Ranking frames by quality ...")
    t0 = time.time()
    quality_scores, sorted_indices = rank_frames(
        frames_data,
        method=params["ranking_method"],
        normalize=params["normalize_brightness"],
        stride=params["ranking_stride"],
    )
    best_idx = sorted_indices[0]
    print(f"       Best frame: #{best_idx} "
          f"(score={quality_scores[best_idx]:.4f}), done in {time.time()-t0:.1f}s")

    # ── [4/8] Global alignment ───────────────────────────────────────────
    print("[4/8] Global alignment ...")
    t0 = time.time()
    best_blurred = frames_data['blurred'][best_idx]
    rect = find_alignment_rect(best_blurred)
    shifts, intersection, mean_frame = align_frames_global(
        frames_data, sorted_indices, rect,
        search_width=params["alignment_search_width"],
        average_frame_percent=params["average_frame_percent"],
    )
    print(f"       Intersection: {intersection}, done in {time.time()-t0:.1f}s")

    # ── [5/8] Create alignment point grid ─────────────────────────────────
    print("[5/8] Creating alignment point grid ...")
    t0 = time.time()
    alignment_points = create_ap_grid(
        mean_frame,
        half_box_width=params["ap_half_box_width"],
        structure_threshold=params["ap_structure_threshold"],
        brightness_threshold=params["ap_brightness_threshold"],
        search_width=params["ap_search_width"],
    )
    print(f"       Created {len(alignment_points)} alignment points "
          f"in {time.time()-t0:.1f}s")

    # ── [6/8] Local frame ranking ─────────────────────────────────────────
    print("[6/8] Ranking frames at each alignment point ...")
    t0 = time.time()
    alignment_points = rank_frames_local(
        frames_data, alignment_points, shifts,
        frame_percent=params["ap_frame_percent"],
        method=params["ranking_method"],
        stride=params["ranking_stride"],
        normalize=params["normalize_brightness"],
    )
    print(f"       Done in {time.time()-t0:.1f}s")

    # ── [7/9] Compute local shifts and stack ──────────────────────────────
    print("[7/9] Computing local shifts ...")
    t0 = time.time()
    alignment_points = compute_local_shifts(
        frames_data, alignment_points, shifts,
        search_width=params["ap_search_width"],
        gauss_width=params["gauss_width"],
    )
    print(f"       Done in {time.time()-t0:.1f}s")

    print("       Stacking and blending ...")
    t0 = time.time()
    stacked_raw = stack_and_blend(
        frames, alignment_points, shifts, intersection,
        mean_frame,
        drizzle_factor=params["drizzle_factor"],
        normalize_brightness=params["normalize_brightness"],
    )
    print(f"       Output shape: {stacked_raw.shape}, dtype: {stacked_raw.dtype}, "
          f"done in {time.time()-t0:.1f}s")

    # ── [8/9] Post-processing: unsharp masking ────────────────────────────
    print("[8/9] Applying unsharp masking ...")
    usm_sigma = params.get("usm_sigma", 2.0)
    usm_alpha = params.get("usm_alpha", 1.5)
    stacked = unsharp_mask(stacked_raw, sigma=usm_sigma, alpha=usm_alpha)
    print(f"       USM sigma={usm_sigma}, alpha={usm_alpha}")

    # ── [9/9] Compute metrics and save outputs ────────────────────────────
    print("[9/9] Computing metrics and saving outputs ...")

    # Compute a simple mean of all frames for comparison
    int_y_lo, int_y_hi, int_x_lo, int_x_hi = intersection
    simple_mean = np.zeros((int_y_hi - int_y_lo, int_x_hi - int_x_lo, 3),
                            dtype=np.float64)
    for i in range(frames.shape[0]):
        dy, dx = int(shifts[i, 0]), int(shifts[i, 1])
        src = frames[i, int_y_lo + dy:int_y_hi + dy,
                       int_x_lo + dx:int_x_hi + dx]
        h_s, w_s = src.shape[:2]
        simple_mean[:h_s, :w_s] += src.astype(np.float64)
    simple_mean = (simple_mean / frames.shape[0]).astype(np.uint8)

    best_frame = frames[best_idx, int_y_lo:int_y_hi, int_x_lo:int_x_hi]

    metrics = compute_metrics(
        stacked, simple_mean, best_frame,
        n_alignment_points=len(alignment_points),
        n_frames_used=frames.shape[0],
    )

    # NCC / NRMSE vs baseline reference (PSS-stacked output in data/)
    ref_stacked = np.load(os.path.join(data_dir, "baseline_reference.npz"))['stacked'][0].astype(np.float64)
    s64 = stacked.astype(np.float64)
    s_flat, r_flat = s64.ravel(), ref_stacked.ravel()
    sc, rc = s_flat - s_flat.mean(), r_flat - r_flat.mean()
    ncc_ref   = float(np.dot(sc, rc) / (np.linalg.norm(sc) * np.linalg.norm(rc) + 1e-12))
    nrmse_ref = float(np.sqrt(np.mean((s_flat - r_flat)**2)) /
                      (r_flat.max() - r_flat.min() + 1e-12))
    metrics['ncc_vs_ref']   = round(ncc_ref, 4)
    metrics['nrmse_vs_ref'] = round(nrmse_ref, 4)

    # Print metrics
    for k, v in metrics.items():
        print(f"       {k}: {v}")

    # Save outputs
    np.save(os.path.join(ref_dir, "stacked.npy"), stacked)
    np.save(os.path.join(ref_dir, "stacked_raw.npy"), stacked_raw)
    np.save(os.path.join(ref_dir, "simple_mean.npy"), simple_mean)
    np.save(os.path.join(ref_dir, "best_frame.npy"), best_frame)
    np.save(os.path.join(ref_dir, "quality_scores.npy"), quality_scores)
    np.save(os.path.join(ref_dir, "sorted_indices.npy"), sorted_indices)
    np.save(os.path.join(ref_dir, "global_shifts.npy"), shifts)

    # Save standard metrics.json to evaluation/ (harness location)
    eval_metrics_path = os.path.join(TASK_DIR, "evaluation", "metrics.json")
    with open(eval_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Keep a copy in reference_outputs/ for backward compatibility
    with open(os.path.join(ref_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n       Results saved to {ref_dir}/")
    print("       Done!")


if __name__ == "__main__":
    main()
