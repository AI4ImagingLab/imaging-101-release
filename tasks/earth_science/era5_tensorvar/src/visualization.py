"""Metrics and plots for the ERA5 Tensor-Var task.

The task targets five geophysical fields, so the standard NCC/NRMSE pair is
computed per channel and reported alongside an area-weighted relative error
that uses the latitude weighting matrix shipped with the dataset.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


CHANNEL_NAMES = ("geopotential", "temperature", "humidity", "wind_u", "wind_v")


def compute_weighted_nrmse_per_channel(
    estimate: np.ndarray,
    reference: np.ndarray,
    lat_weight_matrix: np.ndarray,
) -> np.ndarray:
    """Latitude-weighted relative L2 error per channel, averaged over time.

    estimate, reference : (T, C, H, W) — both in the same (normalised) space
    lat_weight_matrix   : (C, H, W) latitude weighting (broadcast across T)

    Returns an array of shape (C,) with one weighted-NRMSE value per channel.
    """
    T, C = estimate.shape[:2]
    errors = np.empty((T, C), dtype=np.float64)
    for t in range(T):
        true = reference[t]
        est = estimate[t]
        norm = np.linalg.norm((true * lat_weight_matrix).reshape(C, -1), axis=1)
        diff_norm = np.linalg.norm(((est - true) * lat_weight_matrix).reshape(C, -1), axis=1)
        errors[t] = diff_norm / norm
    return errors.mean(axis=0)


def compute_metrics_per_channel(
    estimate: np.ndarray,
    reference: np.ndarray,
    lat_weight_matrix: np.ndarray,
) -> dict:
    """Combined NCC / NRMSE / weighted-NRMSE per channel.

    Returns
    -------
    dict with keys:
        ncc                       : (C,) cosine similarity per channel
        nrmse                     : (C,) range-normalised RMSE per channel
        weighted_nrmse            : (C,) latitude-weighted relative L2 error per channel
        ncc_mean / nrmse_mean / weighted_nrmse_mean : floats
    """
    T, C, H, W = estimate.shape
    ncc = np.empty(C, dtype=np.float64)
    nrmse = np.empty(C, dtype=np.float64)

    for c in range(C):
        est_c = estimate[:, c].reshape(-1)
        ref_c = reference[:, c].reshape(-1)
        denom = np.linalg.norm(est_c) * np.linalg.norm(ref_c) + 1e-30
        ncc[c] = float(np.dot(est_c, ref_c) / denom)
        rng = float(ref_c.max() - ref_c.min()) + 1e-30
        nrmse[c] = float(np.sqrt(np.mean((est_c - ref_c) ** 2)) / rng)

    weighted = compute_weighted_nrmse_per_channel(estimate, reference, lat_weight_matrix)

    return {
        "ncc": ncc,
        "nrmse": nrmse,
        "weighted_nrmse": weighted,
        "ncc_mean": float(ncc.mean()),
        "nrmse_mean": float(nrmse.mean()),
        "weighted_nrmse_mean": float(weighted.mean()),
    }


def metrics_to_jsonable(metrics: dict) -> dict:
    """Round + cast a metrics dict for storage in metrics.json."""
    out = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            out[key] = [round(float(v), 6) for v in value.tolist()]
        elif isinstance(value, (np.floating, float)):
            out[key] = round(float(value), 6)
        else:
            out[key] = value
    return out


def plot_all_channels(
    estimate: np.ndarray,
    reference: np.ndarray,
    metrics: dict | None = None,
    channel_names: Sequence[str] = CHANNEL_NAMES,
):
    """Side-by-side strip plots for every channel.

    The figure has one row of (estimate, reference) pairs per channel and one
    column per assimilation timestep. The reference and estimate share a
    colour scale per channel so the eye can compare absolute values.
    """
    import matplotlib.pyplot as plt

    T, C, H, W = estimate.shape
    fig, axes = plt.subplots(2 * C, T, figsize=(2.0 * T, 2.0 * C))
    if T == 1:
        axes = axes.reshape(2 * C, 1)
    for c in range(C):
        vmin = float(min(reference[:, c].min(), estimate[:, c].min()))
        vmax = float(max(reference[:, c].max(), estimate[:, c].max()))
        for t in range(T):
            ax_est = axes[2 * c, t]
            ax_ref = axes[2 * c + 1, t]
            ax_est.imshow(estimate[t, c].T, cmap="jet", vmin=vmin, vmax=vmax)
            ax_ref.imshow(reference[t, c].T, cmap="jet", vmin=vmin, vmax=vmax)
            ax_est.axis("off")
            ax_ref.axis("off")
            if t == 0:
                ax_est.set_ylabel(f"{channel_names[c]}\nest", fontsize=8)
                ax_ref.set_ylabel("ref", fontsize=8)
            if c == 0:
                ax_est.set_title(f"t={t + 1}", fontsize=8)
    if metrics is not None:
        suptitle = "  ".join(
            f"{name}: wNRMSE={float(metrics['weighted_nrmse'][i]):.4f}"
            for i, name in enumerate(channel_names)
        )
        fig.suptitle(suptitle, fontsize=8)
    fig.tight_layout()
    return fig


def print_metrics_table(metrics: dict, channel_names: Sequence[str] = CHANNEL_NAMES) -> None:
    print(f"{'channel':<14s} {'NCC':>8s} {'NRMSE':>8s} {'wNRMSE':>10s}")
    print("-" * 44)
    for i, name in enumerate(channel_names):
        print(
            f"{name:<14s} "
            f"{float(metrics['ncc'][i]):>8.4f} "
            f"{float(metrics['nrmse'][i]):>8.4f} "
            f"{float(metrics['weighted_nrmse'][i]):>10.4f}"
        )
    print("-" * 44)
    print(
        f"{'mean':<14s} "
        f"{metrics['ncc_mean']:>8.4f} "
        f"{metrics['nrmse_mean']:>8.4f} "
        f"{metrics['weighted_nrmse_mean']:>10.4f}"
    )
