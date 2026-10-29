"""
Generate test fixtures for ct_fan_beam evaluation.

Run from the ct_fan_beam task directory:
    cd ct_fan_beam && python generate_fixtures.py
"""

import os
import sys
import numpy as np

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.physics_model import (
    fan_beam_geometry,
    fan_beam_forward_vectorized,
    fan_beam_backproject,
    fan_beam_fbp,
)


def make_phantom(N):
    """Create a simple 32x32 phantom with a centered disk and an off-center blob."""
    phantom = np.zeros((N, N), dtype=np.float64)
    center = N / 2.0
    yy, xx = np.mgrid[:N, :N]
    # Central disk of radius N/4
    r = np.sqrt((xx - center + 0.5) ** 2 + (yy - center + 0.5) ** 2)
    phantom[r < N / 4] = 1.0
    # Small bright blob offset from center
    r2 = np.sqrt((xx - center + 0.5 - N / 8) ** 2 + (yy - center + 0.5 - N / 8) ** 2)
    phantom[r2 < N / 8] = 2.0
    return phantom


def main():
    N = 32
    n_det = 48
    n_angles = 18
    D_sd = 128.0
    D_dd = 128.0

    print("Creating phantom ...")
    phantom = make_phantom(N)

    print("Computing fan-beam geometry ...")
    geo = fan_beam_geometry(N, n_det, n_angles, D_sd, D_dd, angle_range=2 * np.pi)

    print("Forward projection ...")
    sinogram = fan_beam_forward_vectorized(phantom, geo)

    print("Back-projection ...")
    bp = fan_beam_backproject(sinogram, geo)

    print("FBP reconstruction ...")
    fbp = fan_beam_fbp(sinogram, geo, filter_type='hann', cutoff=0.3)

    # Save fixtures
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "evaluation", "fixtures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "physics_model_fixtures.npz")

    print(f"Saving to {out_path} ...")
    np.savez(
        out_path,
        input_phantom=phantom,
        output_sinogram=sinogram,
        output_backprojection=bp,
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
