#!/usr/bin/env python
"""Generate test fixtures for light_field_microscope.

Creates fixtures in evaluation/fixtures/:
  - config_metadata.json  (copy of data/meta_data.json for test_load_metadata)
  - input_case_arrays.npz (synthetic arrays for metric/profile/plot tests)
"""

import os
import sys
import json
import shutil
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

DATA_DIR = os.path.join(TASK_DIR, "data")
FIXTURES_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

from src.preprocessing import load_metadata
from src.generate_data import (
    make_linepair_object,
    resolve_usaf_configuration,
    compute_shift_perpendicular_um_from_sampling,
)
from src.visualization import (
    normalized_cross_correlation,
    normalized_root_mean_square_error,
    compute_image_metrics,
    compute_center_line_profile,
    extract_line_profile,
    normalize_profile,
)


def main():
    print("Generating fixtures for light_field_microscope ...")

    # --- config_metadata.json ---
    # The test expects load_metadata(FIXTURE_DIR / "config_metadata.json") to work
    # and return the same structure as data/meta_data.json
    src_meta = os.path.join(DATA_DIR, "meta_data.json")
    dst_meta = os.path.join(FIXTURES_DIR, "config_metadata.json")
    shutil.copy2(src_meta, dst_meta)
    print("  [OK] config_metadata.json")

    # --- input_case_arrays.npz ---
    # The tests need: input_target, input_rl, input_observation, input_conventional,
    # input_profile_x_um, input_profile_gt, input_profile_conventional, input_profile_rl

    metadata = load_metadata(src_meta)
    usaf_config = resolve_usaf_configuration(metadata)

    # Create a realistic synthetic target
    tex_size = 15 * 15  # n_lenslets * new_spacing_px = 225
    tex_shape = (tex_size, tex_size)
    # Use 1.0 um as approx tex resolution
    tex_res = (1.0, 1.0)

    target = make_linepair_object(
        tex_shape, tex_res,
        lp_per_mm=usaf_config["line_pairs_per_mm"],
        window_size_um=float(usaf_config.get("support_size_um",
                                              metadata.get("usaf_data", {}).get("supportSizeUm", 80.0))),
    )
    target = target.astype(np.float64)

    # Simulate blurred observation & conventional image
    rng = np.random.default_rng(42)
    # Observation: blurred target + noise
    from scipy.ndimage import gaussian_filter
    observation = gaussian_filter(target, sigma=3.0) + rng.normal(0, 0.01, target.shape)
    observation = np.clip(observation, 0, None)

    # Conventional image: slightly more blurred
    conventional = gaussian_filter(target, sigma=5.0)
    conventional = np.clip(conventional, 0, None)

    # RL reconstruction: moderately deblurred
    rl = gaussian_filter(target, sigma=1.5) + rng.normal(0, 0.005, target.shape)
    rl = np.clip(rl, 0, None)

    # Line profiles
    profile_spec = compute_center_line_profile(target, tex_res_x_um=tex_res[0], margin_vox=5)
    profile_gt = extract_line_profile(target, profile_spec)
    profile_conventional = extract_line_profile(conventional, profile_spec)
    profile_rl = extract_line_profile(rl, profile_spec)
    x_um = profile_spec["x_um"]

    np.savez(os.path.join(FIXTURES_DIR, "input_case_arrays.npz"),
             input_target=target,
             input_rl=rl,
             input_observation=observation,
             input_conventional=conventional,
             input_profile_x_um=x_um,
             input_profile_gt=profile_gt,
             input_profile_conventional=profile_conventional,
             input_profile_rl=profile_rl)
    print("  [OK] input_case_arrays.npz")

    # Verify the fixtures work with the test's expected operations
    ncc = normalized_cross_correlation(rl, target)
    nrmse = normalized_root_mean_square_error(rl, target)
    metrics = compute_image_metrics(rl, target)
    profile_norm = normalize_profile(profile_gt)
    print(f"  Verification: NCC={ncc:.4f}, NRMSE={nrmse:.4f}, "
          f"SSIM={metrics['ssim']:.4f}, profile_max={profile_norm.max():.4f}")

    print("Done! All fixtures saved to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
