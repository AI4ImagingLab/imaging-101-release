"""Generate basic fixtures for eht_black_hole_feature_extraction_dynamic.

Runs the generate_data module to produce a small crescent image and
tests basic physics model components.
"""
import sys
import pathlib
import numpy as np

TASK_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from src.generate_data import generate_simple_crescent_image

FIXTURES_DIR = TASK_DIR / "evaluation" / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Generate a small crescent image
    npix = 32
    fov_uas = 200.0
    diameter_uas = 50.0
    width_uas = 10.0
    asymmetry = 0.3
    pa_deg = 45.0

    image = generate_simple_crescent_image(
        npix, fov_uas, diameter_uas, width_uas, asymmetry, pa_deg
    )

    np.savez(
        FIXTURES_DIR / "basic_pipeline.npz",
        crescent_image=image,
        npix=npix,
        fov_uas=fov_uas,
        diameter_uas=diameter_uas,
        width_uas=width_uas,
        asymmetry=asymmetry,
        pa_deg=pa_deg,
    )
    print(f"Saved fixtures to {FIXTURES_DIR / 'basic_pipeline.npz'}")


if __name__ == "__main__":
    main()
