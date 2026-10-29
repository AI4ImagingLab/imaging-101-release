"""
Light Field Microscope Wave-Model Reconstruction
================================================

Run the compact wave-optics light-field benchmark inspired by Broxton et al.:
  1. Load microscope and target metadata
  2. Build wave-optics forward models for thickness-1 benchmark volumes and a
     five-slice volumetric demonstration
  3. Simulate 2D light-field observations
  4. Compare a conventional microscope baseline against Richardson-Lucy 3D
     deconvolution
  5. Save standardized task data, reference outputs, and evaluation metrics

Usage
-----
    cd tasks/light_field_microscope
    python main.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.generate_data import (
    DEFAULT_RANDOM_SEED,
    add_poisson_noise,
    compute_shift_perpendicular_um_from_sampling,
    make_linepair_object,
    place_object_at_depth,
    resolve_usaf_configuration,
)
from src.physics_model import (
    build_volume_system,
    compute_conventional_image,
    write_wave_model_config,
)
from src.preprocessing import load_metadata
from src.solvers import run_richardson_lucy
from src.visualization import (
    compute_center_line_profile,
    compute_image_metrics,
    compute_volume_slice_energy,
    extract_line_profile,
    normalize_profile,
    normalized_cross_correlation,
    normalized_root_mean_square_error,
    plot_light_field_usaf_comparison,
    plot_volume_reconstruction_demo,
)


TASK_DIR = Path(__file__).resolve().parent
DATA_DIR = TASK_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw_data.npz"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.npz"
META_DATA_PATH = DATA_DIR / "meta_data.json"

EVALUATION_DIR = TASK_DIR / "evaluation"
EVALUATION_METRICS_PATH = EVALUATION_DIR / "metrics.json"
REFERENCE_OUTPUTS_DIR = EVALUATION_DIR / "reference_outputs"
REFERENCE_CASES_PATH = REFERENCE_OUTPUTS_DIR / "cases.npz"
REFERENCE_METRICS_PATH = REFERENCE_OUTPUTS_DIR / "metrics.json"
REFERENCE_BASELINE_PATH = REFERENCE_OUTPUTS_DIR / "baseline_reference.npz"
REFERENCE_COMPARISON_PATH = REFERENCE_OUTPUTS_DIR / "light_field_usaf_comparison.png"
REFERENCE_VOLUME_DEMO_PATH = REFERENCE_OUTPUTS_DIR / "volume_demo.npz"
REFERENCE_VOLUME_DEMO_METRICS_PATH = REFERENCE_OUTPUTS_DIR / "volume_demo_metrics.json"
REFERENCE_VOLUME_DEMO_FIGURE_PATH = REFERENCE_OUTPUTS_DIR / "volume_reconstruction_demo.png"
REFERENCE_CONFIG_PATH = REFERENCE_OUTPUTS_DIR / "lfm_system_config.yaml"

OUTPUT_DIR = TASK_DIR / "output"
OUTPUT_RECONSTRUCTION_PATH = OUTPUT_DIR / "reconstruction.npy"
OUTPUT_VOLUME_DEMO_PATH = OUTPUT_DIR / "volume_demo_reconstruction.npy"

# Solver and optics constants (not exposed in meta_data.json to avoid leaking
# algorithm information to evaluation agents).
_RL_ITERATIONS = 10
_THETA_SAMPLES = 128
_KERNEL_TOL = 0.005


def stack_case_arrays(cases: list[dict], key: str) -> np.ndarray:
    return np.stack([np.asarray(case[key], dtype=np.float32) for case in cases], axis=0)


def stack_case_profiles(cases: list[dict], key: str) -> np.ndarray:
    return np.stack([np.asarray(case["profile"][key], dtype=np.float32) for case in cases], axis=0)


def stack_case_depth_grids(cases: list[dict]) -> np.ndarray:
    return np.stack([np.asarray(case["depths_um"], dtype=np.float32) for case in cases], axis=0)


def build_reference_case_archive(cases: list[dict], target_depths_um: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "target_depths_um": np.asarray(target_depths_um[None, :], dtype=np.float32),
        "reconstruction_depths_um": stack_case_depth_grids(cases)[None, ...],
        "target_images": stack_case_arrays(cases, "gt_image")[None, ...],
        "target_volumes": stack_case_arrays(cases, "gt_volume")[None, ...],
        "light_field_observations": stack_case_arrays(cases, "observation")[None, ...],
        "conventional_images": stack_case_arrays(cases, "conventional_image")[None, ...],
        "rl_reconstructions": stack_case_arrays(cases, "rl_image")[None, ...],
        "rl_volumes": stack_case_arrays(cases, "reconstruction_volume")[None, ...],
        "profile_x_um": stack_case_profiles(cases, "x_um")[None, ...],
        "profiles_gt": stack_case_profiles(cases, "gt")[None, ...],
        "profiles_conventional": stack_case_profiles(cases, "conventional")[None, ...],
        "profiles_rl": stack_case_profiles(cases, "rl")[None, ...],
    }


def save_standardized_data(cases: list[dict], target_depths_um: np.ndarray) -> None:
    np.savez_compressed(
        RAW_DATA_PATH,
        light_field_observations=stack_case_arrays(cases, "observation")[None, ...],
        reconstruction_depths_um=stack_case_depth_grids(cases)[None, ...],
    )
    np.savez_compressed(
        GROUND_TRUTH_PATH,
        target_images=stack_case_arrays(cases, "gt_image")[None, ...],
        target_volumes=stack_case_arrays(cases, "gt_volume")[None, ...],
        target_depths_um=np.asarray(target_depths_um[None, :], dtype=np.float32),
    )


def build_boundary_metrics(reference_reconstruction: np.ndarray, conventional_baseline: np.ndarray) -> dict:
    conventional_flux_matched = conventional_baseline.astype(np.float64) * (
        reference_reconstruction.sum() / (conventional_baseline.sum() + 1e-30)
    )
    baseline_ncc = normalized_cross_correlation(conventional_flux_matched, reference_reconstruction)
    baseline_nrmse = normalized_root_mean_square_error(conventional_flux_matched, reference_reconstruction)
    return {
        "baseline": [
            {
                "method": "conventional microscope baseline",
                "ncc_vs_ref": round(baseline_ncc, 4),
                "nrmse_vs_ref": round(baseline_nrmse, 4),
            }
        ],
        "nrmse_definition": "sqrt(mean((x - x_ref)^2)) / (max(x_ref) - min(x_ref))",
        "ncc_boundary": round(0.9 * baseline_ncc, 4),
        "nrmse_boundary": round(1.1 * baseline_nrmse, 4),
    }


def resolve_volume_demo_configuration(metadata: dict, usaf: dict) -> dict:
    demo = metadata.get("volume_demo", {})
    depth_range_um = np.asarray(demo.get("depthRangeUm", [-100.0, 0.0]), dtype=np.float64)
    if depth_range_um.shape != (2,):
        raise ValueError(f"Expected volume_demo.depthRangeUm to contain two values, got {depth_range_um}.")

    depth_step_um = float(demo.get("depthStepUm", 25.0))
    target_depth_um = float(demo.get("targetDepthUm", -50.0))
    reconstruction_depths_um = np.arange(
        float(depth_range_um[0]),
        float(depth_range_um[1]) + 0.5 * depth_step_um,
        depth_step_um,
        dtype=np.float64,
    )
    return {
        "depth_range_um": (float(depth_range_um[0]), float(depth_range_um[1])),
        "depth_step_um": depth_step_um,
        "target_depth_um": target_depth_um,
        "reconstruction_depths_um": reconstruction_depths_um,
        "line_pairs_per_mm": float(demo.get("linePairsPerMM", usaf["line_pairs_per_mm"])),
        "support_size_um": float(demo.get("supportSizeUm", usaf["support_size_um"])),
        "field_of_view_scale": float(demo.get("fieldOfViewScale", usaf["field_of_view_scale"])),
        "perpendicular_shift_lenslet_pitch": float(
            demo.get("perpendicularShiftLensletPitch", usaf["perpendicular_shift_lenslet_pitch"])
        ),
        "perpendicular_shift_um": (
            None if demo.get("perpendicularShiftUm", usaf["perpendicular_shift_um"]) is None else float(
                demo.get("perpendicularShiftUm", usaf["perpendicular_shift_um"])
            )
        ),
        "background": float(demo.get("background", usaf["background"])),
        "poisson_scale": float(demo.get("poissonScale", usaf["poisson_scale"])),
    }


def main():
    REFERENCE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading microscope and volume-reconstruction configuration...")
    metadata = load_metadata(META_DATA_PATH)
    reconstruction = metadata["reconstruction"]
    usaf = resolve_usaf_configuration(metadata)
    volume_demo = resolve_volume_demo_configuration(metadata, usaf)
    target_depths_um = np.asarray(usaf["target_depths_um"], dtype=np.float64)

    base_n_lenslets = int(metadata["mla"].get("n_lenslets", 15))
    new_spacing_px = int(reconstruction["newSpacingPx"])
    depth_step_um = float(reconstruction["depthStep"])
    theta_samples = _THETA_SAMPLES
    kernel_tol = _KERNEL_TOL
    rl_iterations = _RL_ITERATIONS

    benchmark_n_lenslets = max(1, int(round(base_n_lenslets * float(usaf["field_of_view_scale"]))))
    volume_demo_n_lenslets = max(1, int(round(base_n_lenslets * float(volume_demo["field_of_view_scale"]))))
    benchmark_rng = np.random.default_rng(DEFAULT_RANDOM_SEED) if usaf["poisson_scale"] > 0 else None
    volume_demo_rng = (
        np.random.default_rng(DEFAULT_RANDOM_SEED + 1) if volume_demo["poisson_scale"] > 0 else None
    )

    print(f"  Benchmark depths  : {target_depths_um.tolist()} um")
    print(f"  Demo depth grid   : {volume_demo['reconstruction_depths_um'].tolist()} um")
    print(f"  RL iterations     : {rl_iterations}")
    print(f"  Pixels / lenslet  : {new_spacing_px}")

    print("\nStep 2: Preparing wave-optics system configuration...")
    write_wave_model_config(metadata, REFERENCE_CONFIG_PATH)
    print(f"  Config saved to   : {REFERENCE_CONFIG_PATH}")

    print("\nStep 3: Simulating thickness-1 benchmark volumes...")
    cases: list[dict] = []

    for case_idx, target_depth_um in enumerate(target_depths_um):
        print(f"  Case {case_idx + 1}/{len(target_depths_um)}: occupied slice at z = {target_depth_um:.0f} um")

        system = build_volume_system(
            config_path=REFERENCE_CONFIG_PATH,
            n_lenslets=benchmark_n_lenslets,
            new_spacing_px=new_spacing_px,
            depth_range_um=(float(target_depth_um), float(target_depth_um)),
            depth_step_um=depth_step_um,
            theta_samples=theta_samples,
            kernel_tol=kernel_tol,
        )

        shift_perpendicular_um = compute_shift_perpendicular_um_from_sampling(
            tex_nnum_x=float(system.resolution.tex_nnum[1]),
            tex_res_x_um=float(system.resolution.tex_res[1]),
            lenslet_pitch_fraction=usaf["perpendicular_shift_lenslet_pitch"],
            shift_um=usaf["perpendicular_shift_um"],
        )

        object_2d = make_linepair_object(
            tex_shape=system.tex_shape,
            tex_res_xy_um=(float(system.resolution.tex_res[0]), float(system.resolution.tex_res[1])),
            lp_per_mm=usaf["line_pairs_per_mm"],
            window_size_um=usaf["support_size_um"],
            shift_perpendicular_um=shift_perpendicular_um,
        )
        profile_spec = compute_center_line_profile(
            object_2d,
            tex_res_x_um=float(system.resolution.tex_res[1]),
            margin_vox=int(usaf["profile_margin_vox"]),
        )

        gt_volume = place_object_at_depth(
            object_2d=object_2d,
            tex_shape=system.tex_shape,
            depths=system.depths,
            target_depth_um=float(target_depth_um),
            background=usaf["background"],
        )
        observation_clean = system.forward_project(gt_volume)
        observation = (
            add_poisson_noise(observation_clean, usaf["poisson_scale"], benchmark_rng)
            if benchmark_rng is not None
            else np.asarray(observation_clean, dtype=np.float64).copy()
        )

        conventional_image = compute_conventional_image(
            system=system,
            object_2d=object_2d,
            target_depth_um=float(target_depth_um),
            theta_samples=theta_samples,
        )
        reconstruction_volume = run_richardson_lucy(
            system=system,
            observation=observation,
            iterations=rl_iterations,
            init=np.ones_like(gt_volume),
        )
        rl_image = reconstruction_volume[:, :, 0]

        conventional_metrics = compute_image_metrics(conventional_image, object_2d)
        rl_metrics = compute_image_metrics(rl_image, object_2d)

        cases.append(
            {
                "target_depth_um": float(target_depth_um),
                "depths_um": np.asarray(system.depths, dtype=np.float64),
                "gt_image": np.asarray(object_2d, dtype=np.float64),
                "gt_volume": np.asarray(gt_volume, dtype=np.float64),
                "observation": np.asarray(observation, dtype=np.float64),
                "conventional_image": np.asarray(conventional_image, dtype=np.float64),
                "reconstruction_volume": np.asarray(reconstruction_volume, dtype=np.float64),
                "rl_image": np.asarray(rl_image, dtype=np.float64),
                "conventional": conventional_metrics,
                "rl": rl_metrics,
                "profile": {
                    "row": int(profile_spec["row"]),
                    "x_um": np.asarray(profile_spec["x_um"], dtype=np.float32),
                    "gt": normalize_profile(extract_line_profile(object_2d, profile_spec)),
                    "conventional": normalize_profile(extract_line_profile(conventional_image, profile_spec)),
                    "rl": normalize_profile(extract_line_profile(rl_image, profile_spec)),
                },
            }
        )

    print("\nStep 4: Running generalized five-slice volume reconstruction demo...")
    volume_system = build_volume_system(
        config_path=REFERENCE_CONFIG_PATH,
        n_lenslets=volume_demo_n_lenslets,
        new_spacing_px=new_spacing_px,
        depth_range_um=volume_demo["depth_range_um"],
        depth_step_um=volume_demo["depth_step_um"],
        theta_samples=theta_samples,
        kernel_tol=kernel_tol,
    )
    volume_shift_perpendicular_um = compute_shift_perpendicular_um_from_sampling(
        tex_nnum_x=float(volume_system.resolution.tex_nnum[1]),
        tex_res_x_um=float(volume_system.resolution.tex_res[1]),
        lenslet_pitch_fraction=volume_demo["perpendicular_shift_lenslet_pitch"],
        shift_um=volume_demo["perpendicular_shift_um"],
    )
    volume_object_2d = make_linepair_object(
        tex_shape=volume_system.tex_shape,
        tex_res_xy_um=(float(volume_system.resolution.tex_res[0]), float(volume_system.resolution.tex_res[1])),
        lp_per_mm=volume_demo["line_pairs_per_mm"],
        window_size_um=volume_demo["support_size_um"],
        shift_perpendicular_um=volume_shift_perpendicular_um,
    )
    volume_gt = place_object_at_depth(
        object_2d=volume_object_2d,
        tex_shape=volume_system.tex_shape,
        depths=volume_system.depths,
        target_depth_um=volume_demo["target_depth_um"],
        background=volume_demo["background"],
    )
    volume_observation_clean = volume_system.forward_project(volume_gt)
    volume_observation = (
        add_poisson_noise(volume_observation_clean, volume_demo["poisson_scale"], volume_demo_rng)
        if volume_demo_rng is not None
        else np.asarray(volume_observation_clean, dtype=np.float64).copy()
    )
    volume_conventional_image = compute_conventional_image(
        system=volume_system,
        object_2d=volume_object_2d,
        target_depth_um=float(volume_demo["target_depth_um"]),
        theta_samples=theta_samples,
    )
    volume_reconstruction = run_richardson_lucy(
        system=volume_system,
        observation=volume_observation,
        iterations=rl_iterations,
        init=np.ones_like(volume_gt),
    )

    gt_slice_energy = compute_volume_slice_energy(volume_gt)
    rl_slice_energy = compute_volume_slice_energy(volume_reconstruction)
    gt_energy_share = gt_slice_energy / max(float(gt_slice_energy.sum()), 1e-30)
    rl_energy_share = rl_slice_energy / max(float(rl_slice_energy.sum()), 1e-30)
    target_slice_idx = int(np.argmin(np.abs(np.asarray(volume_system.depths) - volume_demo["target_depth_um"])))
    reconstructed_peak_idx = int(np.argmax(rl_slice_energy))
    volume_demo_metrics = {
        "target_depth_um": float(volume_demo["target_depth_um"]),
        "reconstruction_depths_um": [float(v) for v in np.asarray(volume_system.depths)],
        "ground_truth_slice_energy": [float(v) for v in gt_slice_energy],
        "reconstruction_slice_energy": [float(v) for v in rl_slice_energy],
        "ground_truth_energy_share": [float(v) for v in gt_energy_share],
        "reconstruction_energy_share": [float(v) for v in rl_energy_share],
        "target_slice_index": target_slice_idx,
        "reconstructed_peak_index": reconstructed_peak_idx,
        "reconstructed_peak_depth_um": float(np.asarray(volume_system.depths)[reconstructed_peak_idx]),
        "peak_depth_error_um": float(
            np.asarray(volume_system.depths)[reconstructed_peak_idx] - volume_demo["target_depth_um"]
        ),
        "target_slice_ncc": float(
            normalized_cross_correlation(
                volume_reconstruction[:, :, target_slice_idx],
                volume_gt[:, :, target_slice_idx],
            )
        ),
        "target_slice_nrmse": float(
            normalized_root_mean_square_error(
                volume_reconstruction[:, :, target_slice_idx],
                volume_gt[:, :, target_slice_idx],
            )
        ),
        "off_target_energy_fraction": float(1.0 - rl_energy_share[target_slice_idx]),
    }
    volume_demo_archive = {
        "observation": np.asarray(volume_observation[None, ...], dtype=np.float32),
        "conventional_image": np.asarray(volume_conventional_image[None, ...], dtype=np.float32),
        "ground_truth_volume": np.moveaxis(np.asarray(volume_gt, dtype=np.float32), -1, 0)[None, ...],
        "reconstruction_volume": np.moveaxis(np.asarray(volume_reconstruction, dtype=np.float32), -1, 0)[None, ...],
        "depths_um": np.asarray(np.asarray(volume_system.depths)[None, :], dtype=np.float32),
        "target_depth_um": np.asarray([[volume_demo["target_depth_um"]]], dtype=np.float32),
        "ground_truth_energy_share": np.asarray(gt_energy_share[None, :], dtype=np.float32),
        "reconstruction_energy_share": np.asarray(rl_energy_share[None, :], dtype=np.float32),
    }

    print("\nStep 5: Saving standardized task artifacts...")
    save_standardized_data(cases, target_depths_um)
    reference_archive = build_reference_case_archive(cases, target_depths_um)
    np.savez_compressed(REFERENCE_CASES_PATH, **reference_archive)
    np.savez_compressed(REFERENCE_VOLUME_DEMO_PATH, **volume_demo_archive)

    target_images = reference_archive["target_images"][0]
    conventional_images = reference_archive["conventional_images"][0]
    rl_reconstructions = reference_archive["rl_reconstructions"][0]

    np.savez_compressed(REFERENCE_BASELINE_PATH, rl_reconstructions=rl_reconstructions[None, ...].astype(np.float32))
    np.save(OUTPUT_RECONSTRUCTION_PATH, rl_reconstructions.astype(np.float32))
    np.save(OUTPUT_VOLUME_DEMO_PATH, np.asarray(volume_reconstruction, dtype=np.float32))

    comparison_figure = plot_light_field_usaf_comparison(
        cases,
        rl_iterations=rl_iterations,
        title=(
            "Wave-Optics Light Field Microscope Thickness-1 Volume Benchmark\n"
            "line-pair target, conventional baseline, RL reconstruction, and center-line profiles"
        ),
    )
    comparison_figure.savefig(REFERENCE_COMPARISON_PATH, dpi=200, bbox_inches="tight")
    plt.close(comparison_figure)

    volume_demo_figure = plot_volume_reconstruction_demo(
        observation=volume_observation,
        gt_volume=volume_gt,
        reconstruction_volume=volume_reconstruction,
        depths_um=np.asarray(volume_system.depths),
        target_depth_um=float(volume_demo["target_depth_um"]),
        title=(
            "Wave-Optics Light Field Microscope Volume Demo\n"
            "single capture, five reconstruction depths, target placed at z = -50 um"
        ),
    )
    volume_demo_figure.savefig(REFERENCE_VOLUME_DEMO_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(volume_demo_figure)

    detailed_metrics = {
        "reconstruction": {
            "newSpacingPx": new_spacing_px,
            "depthStepUm": depth_step_um,
            "thetaSamples": theta_samples,
            "kernelTol": kernel_tol,
            "rlIterations": rl_iterations,
            "nrmseDefinition": "sqrt(mean((x - x_ref)^2)) / (max(x_ref) - min(x_ref))",
        },
        "usaf": {
            "targetDepthsUm": [float(v) for v in target_depths_um],
            "linePairsPerMM": float(usaf["line_pairs_per_mm"]),
            "supportSizeUm": float(usaf["support_size_um"]),
            "fieldOfViewScale": float(usaf["field_of_view_scale"]),
            "perpendicularShiftLensletPitch": float(usaf["perpendicular_shift_lenslet_pitch"]),
            "perpendicularShiftUm": (
                None if usaf["perpendicular_shift_um"] is None else float(usaf["perpendicular_shift_um"])
            ),
            "profileMarginVox": int(usaf["profile_margin_vox"]),
            "poissonScale": float(usaf["poisson_scale"]),
            "randomSeed": DEFAULT_RANDOM_SEED if benchmark_rng is not None else None,
            "interpretation": "Each benchmark case is a thickness-1 volume whose only occupied slice matches the sample depth.",
        },
        "cases": [
            {
                "target_depth_um": case["target_depth_um"],
                "reconstruction_depths_um": [float(v) for v in case["depths_um"]],
                "conventional": case["conventional"],
                "rl": case["rl"],
            }
            for case in cases
        ],
        "volume_demo": {
            **volume_demo_metrics,
            "poissonScale": float(volume_demo["poisson_scale"]),
            "randomSeed": DEFAULT_RANDOM_SEED + 1 if volume_demo_rng is not None else None,
        },
    }
    boundary_metrics = build_boundary_metrics(
        reference_reconstruction=rl_reconstructions,
        conventional_baseline=conventional_images,
    )

    with REFERENCE_METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(detailed_metrics, handle, indent=2)
    with REFERENCE_VOLUME_DEMO_METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(volume_demo_metrics, handle, indent=2)
    with EVALUATION_METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(boundary_metrics, handle, indent=2)

    print(f"  Raw data              : {RAW_DATA_PATH}")
    print(f"  Ground truth data     : {GROUND_TRUTH_PATH}")
    print(f"  Reference cases       : {REFERENCE_CASES_PATH}")
    print(f"  Benchmark figure      : {REFERENCE_COMPARISON_PATH}")
    print(f"  Volume demo archive   : {REFERENCE_VOLUME_DEMO_PATH}")
    print(f"  Volume demo figure    : {REFERENCE_VOLUME_DEMO_FIGURE_PATH}")
    print(f"  Reference metrics     : {REFERENCE_METRICS_PATH}")
    print(f"  Volume demo metrics   : {REFERENCE_VOLUME_DEMO_METRICS_PATH}")
    print(f"  Eval metrics          : {EVALUATION_METRICS_PATH}")
    print(f"  Output recon          : {OUTPUT_RECONSTRUCTION_PATH}")
    print(f"  Output volume recon   : {OUTPUT_VOLUME_DEMO_PATH}")

    print("\nResults")
    print("=======")
    for case_metrics in detailed_metrics["cases"]:
        print(
            f"z={case_metrics['target_depth_um']:.0f} um | "
            f"Conventional: NRMSE={case_metrics['conventional']['nrmse']:.4f}, "
            f"SSIM={case_metrics['conventional']['ssim']:.4f}, "
            f"PSNR={case_metrics['conventional']['psnr']:.2f} | "
            f"RL: NRMSE={case_metrics['rl']['nrmse']:.4f}, "
            f"SSIM={case_metrics['rl']['ssim']:.4f}, "
            f"PSNR={case_metrics['rl']['psnr']:.2f}"
        )
    print(
        "Volume demo | "
        f"target z={volume_demo_metrics['target_depth_um']:.0f} um, "
        f"reconstructed peak z={volume_demo_metrics['reconstructed_peak_depth_um']:.0f} um, "
        f"off-target energy fraction={volume_demo_metrics['off_target_energy_fraction']:.4f}"
    )

    return cases, detailed_metrics, boundary_metrics, volume_demo_metrics


if __name__ == "__main__":
    main()
