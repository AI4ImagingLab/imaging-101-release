"""Generate basic fixtures for reflection_ODT.

Creates a small phantom and runs the BPM forward model components.
"""
import sys
import pathlib
import numpy as np
import torch

TASK_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from src.physics_model import ReflectionBPMConfig, ReflectionBPMForwardModel

FIXTURES_DIR = TASK_DIR / "evaluation" / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Small config
    metadata = {
        "volume_shape": [4, 16, 16],
        "n0": 1.5,
        "NA_obj": 0.55,
        "wavelength_um": 0.532,
        "res_um": [0.1, 0.1, 0.5],
        "ri_contrast": 0.02,
        "illumination_rings": [
            {"NA": 0.3, "n_angles": 2, "type": "BF"},
        ],
        "dz_layer": 0.5,
        "dz_gap": 10.0,
    }

    config = ReflectionBPMConfig.from_metadata(metadata)
    model = ReflectionBPMForwardModel(config, device="cpu")

    # Small phantom
    rng = np.random.default_rng(42)
    nz, ny, nx = config.volume_shape
    delta_n = rng.uniform(-0.01, 0.01, (nz, ny, nx))
    dn_tensor = torch.tensor(delta_n, dtype=torch.float64)

    # Forward pass
    with torch.no_grad():
        intensity = model.forward(dn_tensor)
    intensity_np = intensity.cpu().numpy()

    np.savez(
        FIXTURES_DIR / "basic_pipeline.npz",
        delta_n=delta_n,
        intensity=intensity_np,
        n_angles=config.n_angles,
    )
    print(f"Saved fixtures to {FIXTURES_DIR / 'basic_pipeline.npz'}")


if __name__ == "__main__":
    main()
