"""Generate basic fixtures for conventional_ptychography.

Runs the physics model functions on small synthetic data to produce
fixture files for testing.
"""
import sys
import pathlib
import numpy as np

TASK_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from src.utils import fft2c, ifft2c, circ, gaussian2D, aspw
from src.physics_model import (
    get_object_patch,
    compute_exit_wave,
    fraunhofer_propagate,
    compute_detector_intensity,
    forward_model,
)

FIXTURES_DIR = TASK_DIR / "evaluation" / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Small synthetic object and probe
    Np = 32
    No = 64
    obj = rng.standard_normal((No, No)) + 1j * rng.standard_normal((No, No))
    probe = rng.standard_normal((Np, Np)) + 1j * rng.standard_normal((Np, Np))

    # Test get_object_patch
    position = (10, 15)
    patch = get_object_patch(obj, position, Np)

    # Test compute_exit_wave
    esw = compute_exit_wave(probe, patch)

    # Test fraunhofer_propagate
    det_field = fraunhofer_propagate(esw)

    # Test compute_detector_intensity
    intensity = compute_detector_intensity(det_field)

    # Test forward_model
    intensity_fm, esw_fm = forward_model(probe, obj, position, Np, propagator="Fraunhofer")

    # Test fft2c roundtrip
    field = rng.standard_normal((Np, Np)) + 1j * rng.standard_normal((Np, Np))
    fft_field = fft2c(field)
    roundtrip = ifft2c(fft_field)

    # Test circ
    x, y = np.meshgrid(np.arange(Np) - Np // 2, np.arange(Np) - Np // 2)
    mask = circ(x.astype(float), y.astype(float), Np * 0.8)

    np.savez(
        FIXTURES_DIR / "basic_pipeline.npz",
        object_patch=patch,
        exit_wave=esw,
        detector_field=det_field,
        intensity=intensity,
        intensity_fm=intensity_fm,
        fft_roundtrip=roundtrip,
        circ_mask=mask,
    )
    print(f"Saved fixtures to {FIXTURES_DIR / 'basic_pipeline.npz'}")


if __name__ == "__main__":
    main()
