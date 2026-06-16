"""Generate basic fixtures for lensless_imaging.

Runs the physics model (RealFFTConvolve2D) on small synthetic data.
"""
import sys
import pathlib
import numpy as np

TASK_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from src.physics_model import RealFFTConvolve2D

FIXTURES_DIR = TASK_DIR / "evaluation" / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Small synthetic PSF and scene (H, W, C)
    H, W, C = 16, 16, 3
    psf = rng.uniform(0, 1, (H, W, C)).astype(np.float64)
    psf /= psf.sum(axis=(0, 1), keepdims=True)  # normalize

    scene = rng.uniform(0, 1, (H, W, C)).astype(np.float64)

    # Build forward model
    model = RealFFTConvolve2D(psf)

    # Test forward
    measurement = model.forward(scene)

    # Test pad + convolve + crop cycle
    padded = model._pad(scene)
    convolved = model.convolve(padded)
    cropped = model._crop(convolved)

    # Test deconvolve (adjoint)
    y_padded = model._pad(measurement)
    adjoint = model._crop(model.deconvolve(y_padded))

    np.savez(
        FIXTURES_DIR / "basic_pipeline.npz",
        psf=psf,
        scene=scene,
        measurement=measurement,
        convolved_cropped=cropped,
        adjoint=adjoint,
        padded_shape=np.array(model.padded_shape),
    )
    print(f"Saved fixtures to {FIXTURES_DIR / 'basic_pipeline.npz'}")


if __name__ == "__main__":
    main()
