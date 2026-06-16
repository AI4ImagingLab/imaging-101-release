"""
Visualization Helpers for the Light-Field Microscope Task
=========================================================

These plotting and metric utilities support the current wave-optics USAF
evaluation in `main.py` and the task notebook. Backend selection remains the
responsibility of `main.py` or the notebook environment.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def normalized_cross_correlation(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Compute cosine-style NCC between two arrays."""
    est = np.asarray(estimate, dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()
    denom = np.linalg.norm(est) * np.linalg.norm(ref)
    if denom <= 0:
        return 0.0
    return float(np.dot(est, ref) / denom)


def normalized_root_mean_square_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Compute dynamic-range-normalized RMSE used by the evaluation harness."""
    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    ref_range = float(np.max(ref) - np.min(ref))
    if ref_range <= 0:
        ref_range = 1.0
    return float(np.sqrt(np.mean((est - ref) ** 2)) / ref_range)


def compute_image_metrics(image_2d: np.ndarray, gt_2d: np.ndarray) -> dict:
    image_2d = np.asarray(image_2d, dtype=np.float64)
    gt_2d = np.asarray(gt_2d, dtype=np.float64)
    diff = image_2d - gt_2d
    mse = float(np.mean(diff ** 2))
    ref_range = float(np.max(gt_2d) - np.min(gt_2d))
    if ref_range <= 0:
        ref_range = 1.0
    nrmse = float(np.sqrt(mse) / ref_range)
    data_range = float(max(np.max(gt_2d), np.max(image_2d)) - min(np.min(gt_2d), np.min(image_2d)))
    if data_range <= 0:
        data_range = 1.0
    ssim = float(structural_similarity(gt_2d, image_2d, data_range=data_range))
    psnr = float(peak_signal_noise_ratio(gt_2d, image_2d, data_range=data_range))
    return {
        "mse": mse,
        "nrmse": nrmse,
        "ssim": ssim,
        "psnr": psnr,
    }


def compute_center_line_profile(
    object_2d: np.ndarray,
    tex_res_x_um: float,
    margin_vox: int = 10,
) -> dict:
    center_row = int(object_2d.shape[0] // 2)
    nonzero_cols = np.flatnonzero(object_2d[center_row] > 0)
    if nonzero_cols.size == 0:
        raise ValueError("Center row does not intersect the line-pair target.")

    x0 = max(int(nonzero_cols.min()) - int(margin_vox), 0)
    x1 = min(int(nonzero_cols.max()) + int(margin_vox) + 1, object_2d.shape[1])
    sample_cols = np.arange(x0, x1, dtype=np.int32)
    center_col = object_2d.shape[1] // 2
    x_um = (sample_cols - center_col) * float(tex_res_x_um)

    return {
        "row": center_row,
        "x0": x0,
        "x1": x1,
        "x_um": x_um.astype(np.float32),
    }


def extract_line_profile(image_2d: np.ndarray, profile_spec: dict) -> np.ndarray:
    row = int(profile_spec["row"])
    x0 = int(profile_spec["x0"])
    x1 = int(profile_spec["x1"])
    return np.asarray(image_2d[row, x0:x1], dtype=np.float32)


def normalize_profile(profile: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float32)
    peak = float(np.max(profile))
    if peak <= 0:
        return np.zeros_like(profile)
    return profile / peak


def compute_volume_slice_energy(volume: np.ndarray) -> np.ndarray:
    """Sum the reconstructed energy in each axial slice of a volume."""
    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume with shape (Y, X, Z), got {volume.shape}.")
    return np.sum(np.clip(volume, 0.0, None), axis=(0, 1)).astype(np.float64)


def _normalize_image_for_display(image_2d: np.ndarray) -> tuple[np.ndarray, float]:
    image_2d = np.asarray(image_2d, dtype=np.float64)
    peak = float(np.max(image_2d))
    if peak <= 0:
        return np.zeros_like(image_2d), 0.0
    return image_2d / peak, peak


def plot_light_field_usaf_comparison(cases: list, rl_iterations: int, title: str) -> plt.Figure:
    fig, axes = plt.subplots(
        len(cases),
        5,
        figsize=(21.0, 3.8 * len(cases)),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.0, 1.35]},
        squeeze=False,
    )

    col_titles = [
        "Ground Truth",
        "Light Field Observation",
        "Conventional Microscope",
        f"RL Reconstruction\n({rl_iterations} iterations)",
        "Line Profile\n(center row, across stripes)",
    ]

    for row_idx, case in enumerate(cases):
        target_depth = case["target_depth_um"]
        gt_disp, gt_peak = _normalize_image_for_display(case["gt_image"])
        obs_disp, obs_peak = _normalize_image_for_display(case["observation"])
        conv_disp, conv_peak = _normalize_image_for_display(case["conventional_image"])
        rl_disp, rl_peak = _normalize_image_for_display(case["rl_image"])
        profile_row = int(case["profile"]["row"])

        image_panels = [
            ("Ground Truth", gt_disp, gt_peak, None),
            ("Light Field Observation", obs_disp, obs_peak, None),
            ("Conventional Microscope", conv_disp, conv_peak, case["conventional"]),
            ("RL Reconstruction", rl_disp, rl_peak, case["rl"]),
        ]

        for col_idx, (_, image, peak, metrics) in enumerate(image_panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap="inferno", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.axhline(profile_row, color="cyan", linestyle="--", linewidth=1.0, alpha=0.9)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"z = {target_depth:.0f} um", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

            note_lines = [f"peak={peak:.3g}"]
            if metrics is not None:
                note_lines.extend(
                    [
                        f"MSE={metrics['mse']:.4f}",
                        f"SSIM={metrics['ssim']:.3f}",
                        f"PSNR={metrics['psnr']:.2f}",
                    ]
                )
            ax.text(
                0.02,
                0.03,
                "\n".join(note_lines),
                transform=ax.transAxes,
                color="white",
                fontsize=8.5,
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.55},
            )

        ax = axes[row_idx, 4]
        profile = case["profile"]
        ax.plot(profile["x_um"], profile["gt"], color="black", linewidth=1.8, label="GT")
        ax.plot(profile["x_um"], profile["conventional"], color="tab:orange", linewidth=1.6, label="Conventional")
        ax.plot(profile["x_um"], profile["rl"], color="tab:blue", linewidth=1.6, label="RL")
        ax.set_xlim(float(profile["x_um"][0]), float(profile["x_um"][-1]))
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        if row_idx == 0:
            ax.set_title(col_titles[4], fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.set_xlabel("X (um)", fontsize=9)
        ax.set_ylabel("Normalized intensity", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig


def plot_volume_reconstruction_demo(
    observation: np.ndarray,
    gt_volume: np.ndarray,
    reconstruction_volume: np.ndarray,
    depths_um: np.ndarray,
    target_depth_um: float,
    title: str,
) -> plt.Figure:
    """Plot one observation, all axial slices, and the reconstructed axial energy."""
    depths_um = np.asarray(depths_um, dtype=np.float64)
    gt_volume = np.asarray(gt_volume, dtype=np.float64)
    reconstruction_volume = np.asarray(reconstruction_volume, dtype=np.float64)
    gt_energy = compute_volume_slice_energy(gt_volume)
    reconstruction_energy = compute_volume_slice_energy(reconstruction_volume)
    gt_energy_norm = gt_energy / max(float(gt_energy.sum()), 1e-30)
    reconstruction_energy_norm = reconstruction_energy / max(float(reconstruction_energy.sum()), 1e-30)

    n_depths = int(depths_um.size)
    fig, axes = plt.subplots(
        2,
        n_depths + 1,
        figsize=(3.3 * (n_depths + 1), 6.8),
        gridspec_kw={"width_ratios": [1.1] + [1.0] * n_depths},
        squeeze=False,
    )

    obs_disp, obs_peak = _normalize_image_for_display(observation)
    ax = axes[0, 0]
    ax.imshow(obs_disp, cmap="inferno", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title("Light Field\nObservation", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.02,
        0.03,
        f"peak={obs_peak:.3g}",
        transform=ax.transAxes,
        color="white",
        fontsize=8.5,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.55},
    )

    ax = axes[1, 0]
    ax.plot(depths_um, gt_energy_norm, color="black", linewidth=1.8, marker="o", label="GT energy share")
    ax.plot(
        depths_um,
        reconstruction_energy_norm,
        color="tab:blue",
        linewidth=1.8,
        marker="o",
        label="RL energy share",
    )
    ax.axvline(float(target_depth_um), color="tab:red", linestyle="--", linewidth=1.2, label="Target depth")
    ax.set_xlabel("Depth (um)", fontsize=9)
    ax.set_ylabel("Normalized slice energy", fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    for idx, depth_um in enumerate(depths_um):
        gt_slice = gt_volume[:, :, idx]
        reconstruction_slice = reconstruction_volume[:, :, idx]
        gt_disp, gt_peak = _normalize_image_for_display(gt_slice)
        rec_disp, rec_peak = _normalize_image_for_display(reconstruction_slice)

        ax = axes[0, idx + 1]
        ax.imshow(gt_disp, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"GT z={depth_um:.0f} um", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.02,
            0.03,
            f"share={gt_energy_norm[idx]:.2f}\npeak={gt_peak:.3g}",
            transform=ax.transAxes,
            color="white",
            fontsize=8.5,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.55},
        )

        ax = axes[1, idx + 1]
        ax.imshow(rec_disp, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"RL z={depth_um:.0f} um", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.02,
            0.03,
            f"share={reconstruction_energy_norm[idx]:.2f}\npeak={rec_peak:.3g}",
            transform=ax.transAxes,
            color="white",
            fontsize=8.5,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.55},
        )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig
