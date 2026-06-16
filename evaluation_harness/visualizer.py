"""Evaluation visualization module.

Generates publication-quality comparison figures for end-to-end evaluations.
All figures are saved to the results directory alongside the JSON output,
and their paths are recorded in ``EvalResult.visualization_paths``.

Supports:
  - Side-by-side reconstruction vs ground truth
  - Residual / difference maps
  - Metrics bar charts (for multi-run comparisons)
  - Automated figure directory management

Usage inside the evaluation harness::

    from .visualizer import generate_eval_figures

    fig_paths = generate_eval_figures(
        reconstruction=recon_array,
        ground_truth=gt_array,
        metrics={"nrmse": 0.74, "ncc": 0.70, ...},
        output_dir=Path("results/figures/eht_black_hole_original"),
        run_label="multi_agent_gemini-2.5-pro",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import matplotlib.  If unavailable (e.g. headless CI), we degrade
# gracefully and return empty paths.
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend for headless rendering
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    log.warning("matplotlib not found — visualization will be skipped")


# ===================================================================
# Public API
# ===================================================================


def generate_eval_figures(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    metrics: dict[str, Any],
    output_dir: Path,
    run_label: str = "",
    task_name: str = "",
    pixel_size_uas: float | None = None,
    observation: np.ndarray | None = None,
) -> dict[str, str]:
    """Generate all evaluation figures for one run.

    Parameters
    ----------
    reconstruction : ndarray
        Agent's reconstruction (2-D image, already loaded).
    ground_truth : ndarray
        Reference ground truth image (same shape).
    metrics : dict
        Quality metrics dict from scorer (nrmse, ncc, psnr, ssim, mse).
    output_dir : Path
        Directory where figures will be saved.
    run_label : str
        Human-readable label for this run, e.g. ``"multi_agent_gemini-2.5-pro"``.
    task_name : str
        Task name for figure titles.
    pixel_size_uas : float | None
        If provided, axes are labeled in micro-arcseconds.
    observation : ndarray | None
        Observation image (e.g. dirty image) for the first panel. If None,
        the function will attempt to compute a dirty image from the task data.

    Returns
    -------
    dict mapping figure name → absolute path string.
    """
    if not HAS_MPL:
        log.warning("Skipping visualization (matplotlib not available)")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flux-normalize reconstruction to match GT total flux
    recon = reconstruction.astype(np.float64).copy()
    gt = ground_truth.astype(np.float64).copy()
    recon *= gt.sum() / (recon.sum() + 1e-30)

    paths: dict[str, str] = {}

    # --- 1. Side-by-side comparison: Observation | Ground Truth | Reconstruction ---
    p = _plot_comparison(recon, gt, metrics, output_dir, run_label, task_name, pixel_size_uas, observation=observation)
    if p:
        paths["comparison"] = str(p)

    # --- 2. Residual map ---
    p = _plot_residual(recon, gt, output_dir, run_label, task_name, pixel_size_uas)
    if p:
        paths["residual"] = str(p)

    # --- 3. Metrics summary card ---
    p = _plot_metrics_card(metrics, output_dir, run_label, task_name)
    if p:
        paths["metrics_card"] = str(p)

    # --- 4. Cross-section profile ---
    p = _plot_cross_section(recon, gt, output_dir, run_label, task_name, pixel_size_uas)
    if p:
        paths["cross_section"] = str(p)

    log.info("Generated %d figures in %s", len(paths), output_dir)
    return paths


def generate_comparison_chart(
    results: list[dict[str, Any]],
    output_dir: Path,
    task_name: str = "",
) -> dict[str, str]:
    """Generate bar-chart comparison of multiple runs.

    Parameters
    ----------
    results : list of dict
        Each dict should have keys: ``label``, ``nrmse``, ``ncc``, ``psnr``,
        ``ssim``, ``wall_time_seconds``, ``llm_calls``.
    output_dir : Path
        Where to save the figure.
    task_name : str
        For figure titles.

    Returns
    -------
    dict mapping figure name → path string.
    """
    if not HAS_MPL or not results:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    p = _plot_multi_run_comparison(results, output_dir, task_name)
    if p:
        paths["multi_run_comparison"] = str(p)

    return paths


# ===================================================================
# Observation helpers
# ===================================================================


def compute_dirty_image(
    raw_data_path: Path,
    meta_data_path: Path,
    npix: int = 64,
) -> np.ndarray | None:
    """Compute a dirty image from visibility data for the observation panel.

    Parameters
    ----------
    raw_data_path : Path
        Path to raw_data.npz containing ``vis_cal`` / ``vis_corrupt`` and ``uv_coords``.
    meta_data_path : Path
        Path to meta_data JSON containing ``fov_uas`` and ``npix``.
    npix : int
        Number of pixels (overridden by meta_data if available).

    Returns
    -------
    2-D ndarray (dirty image) or None if data is unavailable.
    """
    import json

    try:
        data = np.load(raw_data_path)
        # Try calibrated visibilities first, fall back to corrupt
        if "vis_cal" in data:
            vis = data["vis_cal"]
        elif "vis_corrupt" in data:
            vis = data["vis_corrupt"]
        else:
            log.warning("No visibility data found in %s", raw_data_path)
            return None

        uv = data["uv_coords"]  # shape (N, 2)

        # Load metadata for FOV
        meta = json.loads(meta_data_path.read_text(encoding="utf-8"))
        fov_rad = float(meta.get("fov_uas", 160)) * 4.848136811095e-12  # uas → rad
        npix = int(meta.get("npix", npix))

        # Grid visibilities onto a 2D UV plane and IFFT
        pixel_size_rad = fov_rad / npix
        uv_max = 1.0 / (2.0 * pixel_size_rad)  # Nyquist

        grid = np.zeros((npix, npix), dtype=complex)
        weights = np.zeros((npix, npix), dtype=float)

        for i in range(len(vis)):
            u, v = uv[i]
            # Map to grid indices
            gi = int(np.round(u / uv_max * (npix / 2))) + npix // 2
            gj = int(np.round(v / uv_max * (npix / 2))) + npix // 2
            if 0 <= gi < npix and 0 <= gj < npix:
                grid[gj, gi] += vis[i]
                weights[gj, gi] += 1.0
                # Hermitian conjugate
                gi2 = npix - 1 - gi
                gj2 = npix - 1 - gj
                if 0 <= gi2 < npix and 0 <= gj2 < npix:
                    grid[gj2, gi2] += np.conj(vis[i])
                    weights[gj2, gi2] += 1.0

        # Normalize
        mask = weights > 0
        grid[mask] /= weights[mask]

        # IFFT to image domain
        dirty = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))))
        # Clip negative artifacts
        dirty = np.maximum(dirty, 0)
        return dirty

    except Exception as e:
        log.warning("Failed to compute dirty image: %s", e)
        return None


# ===================================================================
# Internal plotting functions
# ===================================================================


def _img_extent(n: int, pixel_size_uas: float | None):
    """Return extent for imshow (or None)."""
    if pixel_size_uas is None:
        return None
    hw = n * pixel_size_uas / 2
    return [-hw, hw, -hw, hw]


def _axis_labels(ax, pixel_size_uas: float | None):
    """Set axis labels based on whether we have physical units."""
    if pixel_size_uas is not None:
        ax.set_xlabel("Relative RA (μas)")
        ax.set_ylabel("Relative Dec (μas)")
    else:
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")


def _safe_prefix(run_label: str) -> str:
    """Sanitize run label for use in filenames."""
    return run_label.replace("/", "_").replace("\\", "_").replace(" ", "_")


# -------------------------------------------------------------------
# 1. Side-by-side: Observation | Ground Truth | Reconstruction
# -------------------------------------------------------------------
def _plot_comparison(
    recon: np.ndarray,
    gt: np.ndarray,
    metrics: dict,
    output_dir: Path,
    run_label: str,
    task_name: str,
    pixel_size_uas: float | None,
    observation: np.ndarray | None = None,
) -> Path | None:
    try:
        fig = plt.figure(figsize=(18, 5.5))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1], figure=fig, wspace=0.25)

        extent = _img_extent(gt.shape[0], pixel_size_uas)
        vmax = gt.max()

        # --- Panel 1: Observation (dirty image) ---
        ax0 = fig.add_subplot(gs[0])
        if observation is not None:
            obs_disp = observation.astype(np.float64)
            im0 = ax0.imshow(obs_disp, cmap="afmhot", origin="lower", extent=extent)
            ax0.set_title("Observation (Dirty Image)", fontsize=11)
        else:
            # Placeholder: show text indicating no observation available
            ax0.text(0.5, 0.5, "Observation\nnot available",
                     ha="center", va="center", fontsize=14, color="gray",
                     transform=ax0.transAxes)
            ax0.set_title("Observation", fontsize=11)
        _axis_labels(ax0, pixel_size_uas)

        # --- Panel 2: Ground Truth ---
        ax1 = fig.add_subplot(gs[1])
        im1 = ax1.imshow(gt, cmap="afmhot", origin="lower", extent=extent, vmin=0, vmax=vmax)
        ax1.set_title("Ground Truth", fontsize=11)
        _axis_labels(ax1, pixel_size_uas)

        # --- Panel 3: Reconstruction ---
        ax2 = fig.add_subplot(gs[2])
        im2 = ax2.imshow(recon, cmap="afmhot", origin="lower", extent=extent, vmin=0, vmax=vmax)
        # Build metrics subtitle
        m_str = ""
        if metrics:
            parts = []
            for k in ["nrmse", "ncc", "psnr", "ssim"]:
                if k in metrics and metrics[k] is not None:
                    parts.append(f"{k.upper()}={metrics[k]:.3f}")
            m_str = "\n" + "  ".join(parts)
        ax2.set_title(f"Reconstruction{m_str}", fontsize=10)
        _axis_labels(ax2, pixel_size_uas)

        title = f"{task_name} — {run_label}" if task_name else run_label
        fig.suptitle(title, fontsize=13, y=1.02)
        fig.tight_layout()

        prefix = _safe_prefix(run_label)
        path = output_dir / f"{prefix}_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved comparison figure: %s", path)
        return path
    except Exception as e:
        log.warning("Failed to generate comparison figure: %s", e)
        return None


# -------------------------------------------------------------------
# 2. Residual map with histogram
# -------------------------------------------------------------------
def _plot_residual(
    recon: np.ndarray,
    gt: np.ndarray,
    output_dir: Path,
    run_label: str,
    task_name: str,
    pixel_size_uas: float | None,
) -> Path | None:
    try:
        diff = recon - gt
        fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

        extent = _img_extent(gt.shape[0], pixel_size_uas)
        vlim = max(abs(diff.min()), abs(diff.max()))
        im = ax_img.imshow(diff, cmap="RdBu_r", origin="lower", extent=extent, vmin=-vlim, vmax=vlim)
        ax_img.set_title("Residual Map (Recon − GT)")
        _axis_labels(ax_img, pixel_size_uas)
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

        # Histogram of residuals
        ax_hist.hist(diff.ravel(), bins=100, color="steelblue", alpha=0.8, edgecolor="white", linewidth=0.3)
        ax_hist.axvline(0, color="red", linestyle="--", linewidth=1.5, label="zero")
        ax_hist.set_xlabel("Residual value")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_title(f"Residual Distribution (σ={diff.std():.2e})")
        ax_hist.legend()

        title = f"{task_name} — {run_label}" if task_name else run_label
        fig.suptitle(title, fontsize=12, y=1.02)
        fig.tight_layout()

        prefix = _safe_prefix(run_label)
        path = output_dir / f"{prefix}_residual.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved residual figure: %s", path)
        return path
    except Exception as e:
        log.warning("Failed to generate residual figure: %s", e)
        return None


# -------------------------------------------------------------------
# 3. Metrics summary card
# -------------------------------------------------------------------
def _plot_metrics_card(
    metrics: dict,
    output_dir: Path,
    run_label: str,
    task_name: str,
) -> Path | None:
    try:
        # Prepare data
        metric_names = []
        values = []
        colors = []
        # (metric, higher_is_better, display_range)
        metric_defs = [
            ("nrmse", False, (0, 2)),
            ("ncc", True, (0, 1)),
            ("psnr", True, (0, 40)),
            ("ssim", True, (-0.2, 1)),
        ]
        cmap_good = plt.cm.Greens
        cmap_bad = plt.cm.Reds

        for name, higher_better, (lo, hi) in metric_defs:
            if name not in metrics or metrics[name] is None:
                continue
            v = metrics[name]
            metric_names.append(name.upper())
            values.append(v)
            # Normalize to [0, 1] for color
            frac = (v - lo) / (hi - lo + 1e-30)
            frac = max(0, min(1, frac))
            if not higher_better:
                frac = 1 - frac
            colors.append(cmap_good(0.3 + 0.6 * frac))

        if not metric_names:
            return None

        fig, ax = plt.subplots(figsize=(6, max(3, len(metric_names) * 0.8)))
        y_pos = np.arange(len(metric_names))
        bars = ax.barh(y_pos, values, color=colors, edgecolor="gray", linewidth=0.5, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_names, fontsize=12, fontweight="bold")
        ax.invert_yaxis()

        # Add value labels
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=11)

        title = f"Metrics — {task_name} — {run_label}" if task_name else f"Metrics — {run_label}"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Value")
        fig.tight_layout()

        prefix = _safe_prefix(run_label)
        path = output_dir / f"{prefix}_metrics.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved metrics card: %s", path)
        return path
    except Exception as e:
        log.warning("Failed to generate metrics card: %s", e)
        return None


# -------------------------------------------------------------------
# 4. Cross-section profile
# -------------------------------------------------------------------
def _plot_cross_section(
    recon: np.ndarray,
    gt: np.ndarray,
    output_dir: Path,
    run_label: str,
    task_name: str,
    pixel_size_uas: float | None,
) -> Path | None:
    try:
        ny, nx = gt.shape
        mid_y = ny // 2

        fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(14, 4.5))

        if pixel_size_uas is not None:
            x_axis = (np.arange(nx) - nx / 2) * pixel_size_uas
            xlabel = "Relative RA (μas)"
            y_axis = (np.arange(ny) - ny / 2) * pixel_size_uas
            ylabel = "Relative Dec (μas)"
        else:
            x_axis = np.arange(nx)
            xlabel = "Pixel"
            y_axis = np.arange(ny)
            ylabel = "Pixel"

        # Horizontal cross-section (through middle row)
        ax_h.plot(x_axis, gt[mid_y, :], "k-", linewidth=1.5, label="Ground Truth", alpha=0.8)
        ax_h.plot(x_axis, recon[mid_y, :], "r--", linewidth=1.5, label="Reconstruction", alpha=0.8)
        ax_h.fill_between(x_axis, gt[mid_y, :], recon[mid_y, :], alpha=0.15, color="red")
        ax_h.set_xlabel(xlabel)
        ax_h.set_ylabel("Intensity")
        ax_h.set_title(f"Horizontal Profile (row {mid_y})")
        ax_h.legend(fontsize=9)
        ax_h.grid(True, alpha=0.3)

        # Vertical cross-section (through middle column)
        mid_x = nx // 2
        ax_v.plot(y_axis, gt[:, mid_x], "k-", linewidth=1.5, label="Ground Truth", alpha=0.8)
        ax_v.plot(y_axis, recon[:, mid_x], "r--", linewidth=1.5, label="Reconstruction", alpha=0.8)
        ax_v.fill_between(y_axis, gt[:, mid_x], recon[:, mid_x], alpha=0.15, color="red")
        ax_v.set_xlabel(ylabel)
        ax_v.set_ylabel("Intensity")
        ax_v.set_title(f"Vertical Profile (col {mid_x})")
        ax_v.legend(fontsize=9)
        ax_v.grid(True, alpha=0.3)

        title = f"{task_name} — {run_label}" if task_name else run_label
        fig.suptitle(title, fontsize=12, y=1.02)
        fig.tight_layout()

        prefix = _safe_prefix(run_label)
        path = output_dir / f"{prefix}_cross_section.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved cross-section figure: %s", path)
        return path
    except Exception as e:
        log.warning("Failed to generate cross-section figure: %s", e)
        return None


# -------------------------------------------------------------------
# 5. Multi-run comparison bar chart
# -------------------------------------------------------------------
def _plot_multi_run_comparison(
    results: list[dict[str, Any]],
    output_dir: Path,
    task_name: str,
) -> Path | None:
    """Bar chart comparing multiple evaluation runs across metrics."""
    try:
        labels = [r.get("label", f"run-{i}") for i, r in enumerate(results)]
        metric_keys = ["nrmse", "ncc", "psnr", "ssim"]
        present_metrics = [k for k in metric_keys if any(k in r for r in results)]

        if not present_metrics:
            return None

        n_metrics = len(present_metrics)
        n_runs = len(results)
        x = np.arange(n_metrics)
        width = 0.8 / n_runs

        # Color palette
        colors = plt.cm.Set2(np.linspace(0, 1, max(n_runs, 3)))

        fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_metrics), 5))

        for i, (result, label) in enumerate(zip(results, labels)):
            vals = [result.get(k, 0) for k in present_metrics]
            offset = (i - n_runs / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=label, color=colors[i],
                          edgecolor="gray", linewidth=0.5)
            # Value labels on bars
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([k.upper() for k in present_metrics], fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        title = f"Framework Comparison — {task_name}" if task_name else "Framework Comparison"
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("Metric Value")
        fig.tight_layout()

        path = output_dir / "multi_run_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved multi-run comparison: %s", path)
        return path
    except Exception as e:
        log.warning("Failed to generate multi-run comparison: %s", e)
        return None
