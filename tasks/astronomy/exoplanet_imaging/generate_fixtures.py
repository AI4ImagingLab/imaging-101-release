"""Generate basic fixtures for exoplanet_imaging.

Runs basic physics model and preprocessing functions on small synthetic data.
"""
import sys
import pathlib
import numpy as np
import torch

TASK_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from src.physics_model import rotate_frames, compute_kl_basis_svd
from src.preprocessing import create_circular_mask, mean_subtract_frames

FIXTURES_DIR = TASK_DIR / "evaluation" / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Small synthetic image cube
    N, H, W = 5, 32, 32
    cube = rng.standard_normal((N, H, W)).astype(np.float32)
    angles = np.linspace(-10, 10, N).astype(np.float32)

    # Test mean_subtract_frames
    cube_ms = mean_subtract_frames(cube)

    # Test create_circular_mask
    center = (W // 2, H // 2)
    iwa = 3.0
    owa = 14.0
    mask = create_circular_mask(H, W, center, iwa, owa)

    # Test rotate_frames
    images_t = torch.from_numpy(cube[np.newaxis]).float()  # (1, N, H, W)
    angles_t = torch.from_numpy(angles)
    rotated = rotate_frames(images_t, angles_t)

    # Test KL basis
    flat = torch.from_numpy(cube_ms.reshape(N, -1)).float()
    flat = torch.nan_to_num(flat)
    flat = flat - flat.mean(dim=0, keepdim=True)
    K_max = min(3, N - 1)
    basis = compute_kl_basis_svd(flat, K_max)

    np.savez(
        FIXTURES_DIR / "basic_pipeline.npz",
        cube=cube,
        angles=angles,
        cube_ms=cube_ms,
        mask=mask.numpy(),
        rotated=rotated.numpy(),
        kl_basis=basis.numpy(),
    )
    print(f"Saved fixtures to {FIXTURES_DIR / 'basic_pipeline.npz'}")


if __name__ == "__main__":
    main()
