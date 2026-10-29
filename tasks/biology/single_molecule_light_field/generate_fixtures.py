"""Generate basic fixtures for single_molecule_light_field.

Runs the physics model functions (build_microscope, build_mla) and
preprocessing functions on small synthetic data.
"""
import sys
import pathlib
import numpy as np

TASK_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from src.physics_model import build_microscope, FourierMicroscope
from src.preprocessing import center_localizations

FIXTURES_DIR = TASK_DIR / "evaluation" / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Minimal microscope metadata
    meta = {
        "num_aperture": 1.27,
        "mla_lens_pitch": 222.0,
        "focal_length_mla": 5.556,
        "focal_length_obj_lens": 3.333,
        "focal_length_tube_lens": 200.0,
        "focal_length_fourier_lens": 150.0,
        "pixel_size_camera": 6.5,
        "ref_idx_immersion": 1.406,
        "ref_idx_medium": 1.33,
    }

    # Build microscope
    scope = build_microscope(meta)

    # Test center_localizations with synthetic data
    rng = np.random.default_rng(42)
    n_locs = 50
    locs_2d = np.zeros((n_locs, 8))
    locs_2d[:, 0] = rng.integers(1, 100, n_locs)  # frame
    locs_2d[:, 1] = rng.uniform(-10, 10, n_locs)  # X
    locs_2d[:, 2] = rng.uniform(-10, 10, n_locs)  # Y
    locs_2d[:, 3] = rng.uniform(0.05, 0.2, n_locs)  # sigma_X
    locs_2d[:, 4] = rng.uniform(0.05, 0.2, n_locs)  # sigma_Y
    locs_2d[:, 5] = rng.uniform(100, 5000, n_locs)  # intensity
    locs_2d[:, 6] = rng.uniform(10, 100, n_locs)  # background
    locs_2d[:, 7] = rng.uniform(0.01, 0.1, n_locs)  # precision

    centered = center_localizations(locs_2d)

    np.savez(
        FIXTURES_DIR / "basic_pipeline.npz",
        bfp_radius=scope.bfp_radius,
        magnification=scope.magnification,
        pixel_size_sample=scope.pixel_size_sample,
        rho_scaling=scope.rho_scaling,
        locs_2d=locs_2d,
        locs_centered=centered,
    )
    print(f"Saved fixtures to {FIXTURES_DIR / 'basic_pipeline.npz'}")


if __name__ == "__main__":
    main()
