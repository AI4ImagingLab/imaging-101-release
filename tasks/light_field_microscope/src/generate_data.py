"""
Synthetic Line-Pair Helpers for the Light-Field Microscope Task
===============================================================

This task evaluates wave-optics light-field volume reconstruction on a compact
USAF-style line-pair target. The helpers below keep only the phantom, depth
placement, noise model, and metadata parsing needed by `main.py` and the
auxiliary reference scripts in this task directory.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


DEFAULT_RANDOM_SEED = 0
DEFAULT_USAF_TARGET_DEPTH_UM = 0.0
DEFAULT_USAF_SHIFT_PERPENDICULAR_LENSLETS = 0.5


def make_linepair_object(
    tex_shape: tuple[int, int],
    tex_res_xy_um: tuple[float, float],
    lp_per_mm: float,
    window_size_um: float,
    shift_perpendicular_um: float = 0.0,
) -> np.ndarray:
    """
    Build a square-windowed binary line-pair target.

    The stripes vary along X, so `shift_perpendicular_um` shifts the target
    along X relative to the optical axis.
    """
    row_coords = (np.arange(tex_shape[0], dtype=np.float64) - tex_shape[0] // 2) * tex_res_xy_um[0]
    col_coords = (np.arange(tex_shape[1], dtype=np.float64) - tex_shape[1] // 2) * tex_res_xy_um[1]
    yy, xx = np.meshgrid(row_coords, col_coords, indexing="ij")
    xx_shifted = xx - float(shift_perpendicular_um)
    cycles_per_um = float(lp_per_mm) / 1000.0
    grating = (np.sin(2.0 * np.pi * cycles_per_um * xx_shifted) > 0).astype(np.float64)
    window = (
        (np.abs(xx_shifted) <= float(window_size_um) / 2.0)
        & (np.abs(yy) <= float(window_size_um) / 2.0)
    ).astype(np.float64)
    return grating * window


def place_object_at_depth(
    object_2d: np.ndarray,
    tex_shape: tuple[int, int],
    depths: np.ndarray,
    target_depth_um: float,
    background: float = 0.0,
) -> np.ndarray:
    volume = np.full(tex_shape + (len(depths),), float(background), dtype=np.float64)
    depth_idx = int(np.argmin(np.abs(np.asarray(depths) - float(target_depth_um))))
    volume[:, :, depth_idx] = np.asarray(object_2d, dtype=np.float64)
    return volume


def compute_shift_perpendicular_um_from_sampling(
    tex_nnum_x: float,
    tex_res_x_um: float,
    lenslet_pitch_fraction: float = DEFAULT_USAF_SHIFT_PERPENDICULAR_LENSLETS,
    shift_um: Optional[float] = None,
) -> float:
    if shift_um is not None:
        return float(shift_um)
    return float(lenslet_pitch_fraction) * float(tex_nnum_x) * float(tex_res_x_um)


def resolve_usaf_configuration(metadata: dict) -> dict:
    usaf = metadata.get("usaf_data", {})
    return {
        "target_depths_um": np.asarray(
            usaf.get("targetDepthsUm", usaf.get("target_depths_um", [DEFAULT_USAF_TARGET_DEPTH_UM])),
            dtype=np.float64,
        ),
        "line_pairs_per_mm": float(usaf.get("linePairsPerMM", usaf.get("line_pairs_per_mm", 64.0))),
        "support_size_um": float(usaf.get("supportSizeUm", usaf.get("windowSizeUm", 80.0))),
        "field_of_view_scale": float(usaf.get("fieldOfViewScale", usaf.get("paddingScale", 1.0))),
        "perpendicular_shift_lenslet_pitch": float(
            usaf.get(
                "perpendicularShiftLensletPitch",
                usaf.get(
                    "perpendicular_shift_lenslet_pitch",
                    DEFAULT_USAF_SHIFT_PERPENDICULAR_LENSLETS,
                ),
            )
        ),
        "perpendicular_shift_um": (
            None
            if usaf.get("perpendicularShiftUm", usaf.get("perpendicular_shift_um")) is None
            else float(usaf.get("perpendicularShiftUm", usaf.get("perpendicular_shift_um")))
        ),
        "background": float(usaf.get("background", 0.0)),
        "poisson_scale": float(usaf.get("poissonScale", usaf.get("poisson_scale", 0.0))),
        "profile_margin_vox": int(usaf.get("profileMarginVox", usaf.get("profile_margin_vox", 10))),
    }


def add_poisson_noise(
    image: np.ndarray,
    scale: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Match the compact reference noise model:
    normalize by peak, sample Poisson counts, renormalize, then scale back.
    """
    if rng is None:
        rng = np.random.default_rng()

    image = np.asarray(image, dtype=np.float64)
    image = np.clip(image, 0.0, None)
    peak = float(image.max())
    if peak <= 0 or scale <= 0:
        return image.astype(np.float64)

    image_norm = image / peak
    counts = rng.poisson((image_norm * scale).astype(np.float64)).astype(np.float64)
    count_peak = float(counts.max())
    if count_peak > 0:
        counts /= count_peak
    return counts * peak
