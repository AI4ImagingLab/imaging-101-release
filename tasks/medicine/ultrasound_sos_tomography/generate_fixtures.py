"""
Generate evaluation fixtures for ultrasound_sos_tomography task.

Run from the task directory:
    cd ultrasound_sos_tomography
    python generate_fixtures.py
"""

import os
import sys
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from src.physics_model import radon_forward, filtered_back_projection
from src.visualization import compute_ncc, compute_nrmse, centre_crop

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)


def gen_physics_model_fixtures():
    """Fixture: physics_model_fixtures.npz

    Keys: input_image, input_angles, output_sinogram, output_fbp

    Tests expect:
      - radon_forward(input_image, input_angles).shape == (32, 10)
      - radon_forward values match output_sinogram
      - filtered_back_projection(output_sinogram, input_angles, output_size=32) matches output_fbp
    """
    rng = np.random.default_rng(42)

    # 32x32 image with a simple disc phantom
    N = 32
    x, y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    input_image = np.zeros((N, N), dtype=np.float64)
    input_image[(x ** 2 + y ** 2) < 0.5 ** 2] = 1e-5  # disc

    # 10 angles
    input_angles = np.linspace(0, 180, 10, endpoint=False)

    output_sinogram = radon_forward(input_image, input_angles)
    output_fbp = filtered_back_projection(output_sinogram, input_angles, output_size=N)

    path = os.path.join(FIXTURE_DIR, "physics_model_fixtures.npz")
    np.savez(path,
             input_image=input_image,
             input_angles=input_angles,
             output_sinogram=output_sinogram,
             output_fbp=output_fbp)
    print(f"  Saved {path}")
    print(f"    sinogram shape={output_sinogram.shape}, fbp shape={output_fbp.shape}")


def gen_visualization_fixtures():
    """Fixture: visualization_fixtures.npz

    Keys: input_a, input_b, output_ncc, output_nrmse, output_crop

    Tests expect:
      - compute_ncc(input_a, input_b) matches output_ncc
      - compute_nrmse(input_a, input_b) matches output_nrmse
      - centre_crop(input_a, 0.5) matches output_crop
    """
    rng = np.random.default_rng(42)

    # Use 16x16 arrays (test_ncc_identical uses 16x16)
    input_a = rng.random((16, 16))
    input_b = rng.random((16, 16))

    output_ncc = compute_ncc(input_a, input_b)
    output_nrmse = compute_nrmse(input_a, input_b)
    output_crop = centre_crop(input_a, 0.5)

    path = os.path.join(FIXTURE_DIR, "visualization_fixtures.npz")
    np.savez(path,
             input_a=input_a,
             input_b=input_b,
             output_ncc=np.float64(output_ncc),
             output_nrmse=np.float64(output_nrmse),
             output_crop=output_crop)
    print(f"  Saved {path}")
    print(f"    ncc={output_ncc:.6f}, nrmse={output_nrmse:.6f}, crop shape={output_crop.shape}")


if __name__ == "__main__":
    print("Generating ultrasound_sos_tomography fixtures...")
    gen_physics_model_fixtures()
    gen_visualization_fixtures()
    print("Done!")
