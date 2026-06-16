"""
Generate test fixtures for ct_sparse_view evaluation.

Run from the ct_sparse_view task directory:
    cd ct_sparse_view && python generate_fixtures.py
"""

import os
import sys
import numpy as np

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.physics_model import radon_transform, filtered_back_projection


def make_phantom(N):
    """Create a simple NxN phantom with a disk and a rectangle."""
    phantom = np.zeros((N, N), dtype=np.float64)
    center = N / 2.0
    yy, xx = np.mgrid[:N, :N]
    r = np.sqrt((xx - center + 0.5) ** 2 + (yy - center + 0.5) ** 2)
    # Disk
    phantom[r < N / 3] = 1.0
    # Small bright rectangle
    phantom[N // 4 : N // 4 + N // 8, N // 3 : N // 3 + N // 6] = 2.0
    return phantom


def main():
    N = 32
    n_angles = 18
    angles = np.linspace(0, 180, n_angles, endpoint=False)

    print("Creating phantom ...")
    phantom = make_phantom(N)

    print("Radon transform (forward projection) ...")
    sinogram = radon_transform(phantom, angles)

    print("Filtered back projection ...")
    fbp = filtered_back_projection(sinogram, angles, output_size=N)

    # Save fixtures
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "evaluation", "fixtures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "physics_model_fixtures.npz")

    print(f"Saving to {out_path} ...")
    np.savez(
        out_path,
        input_image=phantom,
        input_angles=angles,
        output_sinogram=sinogram,
        output_fbp=fbp,
    )

    # Verify
    data = np.load(out_path)
    print("Keys:", list(data.keys()))
    for k in data:
        print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")
    print("Done.")


if __name__ == "__main__":
    main()
