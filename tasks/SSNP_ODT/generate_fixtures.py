"""Generate basic fixtures for SSNP_ODT by running core components on small data.

SSNP_ODT uses PyTorch-based forward models. We create fixtures by building
a small config and running the physics model components.
"""
import sys
import pathlib
import numpy as np
import torch

TASK_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from src.physics_model import SSNPConfig, SSNPForwardModel

FIXTURES_DIR = TASK_DIR / "evaluation" / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Small config
    config = SSNPConfig(
        volume_shape=(4, 16, 16),
        res=(0.3, 0.3, 0.5),
        n0=1.33,
        NA=0.55,
        wavelength_um=0.532,
        res_um=(0.1, 0.1, 0.2),
        n_angles=3,
    )

    model = SSNPForwardModel(config, device="cpu")

    # Small phantom (delta_n)
    rng = np.random.default_rng(42)
    nz, ny, nx = config.volume_shape
    delta_n = rng.uniform(-0.01, 0.01, (nz, ny, nx))
    dn_tensor = torch.tensor(delta_n, dtype=torch.float64)

    # Forward pass
    with torch.no_grad():
        intensity = model.forward(dn_tensor)
    intensity_np = intensity.cpu().numpy()

    # Save
    np.savez(
        FIXTURES_DIR / "basic_pipeline.npz",
        delta_n=delta_n,
        intensity=intensity_np,
        kz=model.kz.cpu().numpy(),
        pupil_real=model.pupil.real.cpu().numpy(),
        pupil_imag=model.pupil.imag.cpu().numpy(),
    )
    print(f"Saved fixtures to {FIXTURES_DIR / 'basic_pipeline.npz'}")


if __name__ == "__main__":
    main()
