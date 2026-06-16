"""Generate basic fixtures for lucky_imaging.

Runs physics model functions (quality measures, phase correlation) and
preprocessing functions on small synthetic data.
"""
import sys
import pathlib
import numpy as np

TASK_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from src.physics_model import (
    quality_measure_gradient,
    quality_measure_laplace,
    quality_measure_sobel,
    quality_measure,
    phase_correlation,
    sub_pixel_solve,
)
from src.preprocessing import to_mono, gaussian_blur, average_brightness, compute_laplacian

FIXTURES_DIR = TASK_DIR / "evaluation" / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Small synthetic frame (uint8 RGB)
    H, W = 64, 64
    frame_rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)

    # Preprocessing
    mono = to_mono(frame_rgb)
    blurred = gaussian_blur(mono, gauss_width=7)
    brightness = average_brightness(mono)
    laplacian = compute_laplacian(blurred, stride=2)

    # Quality measures
    frame_u16 = mono.astype(np.uint16) * 256
    qg = quality_measure_gradient(frame_u16, stride=2)
    ql = quality_measure_laplace(blurred, stride=2)
    qs = quality_measure_sobel(mono, stride=2)
    qm = quality_measure(mono.astype(np.float64))

    # Phase correlation
    frame0 = rng.standard_normal((32, 32)).astype(np.float64)
    frame1 = np.roll(np.roll(frame0, 3, axis=0), -2, axis=1)
    shift_y, shift_x = phase_correlation(frame0, frame1, (32, 32))

    # Sub-pixel solve
    vals = rng.standard_normal((3, 3))
    vals[1, 1] = vals.max() + 1.0  # ensure peak at center
    dy, dx = sub_pixel_solve(vals)

    np.savez(
        FIXTURES_DIR / "basic_pipeline.npz",
        mono=mono,
        blurred=blurred,
        brightness=np.array(brightness),
        laplacian=laplacian,
        quality_gradient=np.array(qg),
        quality_laplace=np.array(ql),
        quality_sobel=np.array(qs),
        quality_measure=np.array(qm),
        phase_corr_shift=np.array([shift_y, shift_x]),
        sub_pixel_correction=np.array([dy, dx]),
    )
    print(f"Saved fixtures to {FIXTURES_DIR / 'basic_pipeline.npz'}")


if __name__ == "__main__":
    main()
