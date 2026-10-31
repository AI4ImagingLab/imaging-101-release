"""
Generate evaluation fixtures for pet_mlem task.

Run from the task directory:
    cd pet_mlem
    python generate_fixtures.py
"""

import os
import sys
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from src.generate_data import create_activity_phantom
from src.physics_model import (
    pet_forward_project, pet_back_project,
    compute_sensitivity_image, add_background,
)
from src.preprocessing import preprocess_sinogram

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)


def gen_generate_data_phantom():
    """Fixture: generate_data_phantom.npz

    Keys: param_N, output_phantom
    Test: create_activity_phantom(int(fix['param_N'])) matches fix['output_phantom']
    """
    N = 128
    phantom = create_activity_phantom(N)

    path = os.path.join(FIXTURE_DIR, "generate_data_phantom.npz")
    np.savez(path,
             param_N=np.int64(N),
             output_phantom=phantom)
    print(f"  Saved {path}  phantom shape={phantom.shape} max={phantom.max():.1f}")


def gen_physics_model_forward():
    """Fixture: physics_model_forward.npz

    Keys: input_image, input_theta, output_sinogram
    Test: pet_forward_project(input_image, input_theta) matches output_sinogram
    """
    N = 64
    phantom = create_activity_phantom(N)
    theta = np.linspace(0, 180, 30, endpoint=False)

    sinogram = pet_forward_project(phantom, theta)

    path = os.path.join(FIXTURE_DIR, "physics_model_forward.npz")
    np.savez(path,
             input_image=phantom,
             input_theta=theta,
             output_sinogram=sinogram)
    print(f"  Saved {path}  sinogram shape={sinogram.shape}")


def gen_physics_model_backproject():
    """Fixture: physics_model_backproject.npz

    Keys: input_sinogram, input_theta, param_N, output_image
    Test: pet_back_project(input_sinogram, input_theta, int(param_N)) matches output_image
    """
    N = 64
    phantom = create_activity_phantom(N)
    theta = np.linspace(0, 180, 30, endpoint=False)

    sinogram = pet_forward_project(phantom, theta)
    bp_image = pet_back_project(sinogram, theta, N)

    path = os.path.join(FIXTURE_DIR, "physics_model_backproject.npz")
    np.savez(path,
             input_sinogram=sinogram,
             input_theta=theta,
             param_N=np.int64(N),
             output_image=bp_image)
    print(f"  Saved {path}  bp_image shape={bp_image.shape}")


def gen_physics_model_sensitivity():
    """Fixture: physics_model_sensitivity.npz

    Keys: input_theta, param_N, output_sensitivity
    Test: compute_sensitivity_image(input_theta, int(param_N)) matches output_sensitivity
    """
    N = 64
    theta = np.linspace(0, 180, 30, endpoint=False)

    sensitivity = compute_sensitivity_image(theta, N)

    path = os.path.join(FIXTURE_DIR, "physics_model_sensitivity.npz")
    np.savez(path,
             input_theta=theta,
             param_N=np.int64(N),
             output_sensitivity=sensitivity)
    print(f"  Saved {path}  sensitivity shape={sensitivity.shape}")


def gen_physics_model_background():
    """Fixture: physics_model_background.npz

    Keys: input_sinogram, param_randoms_fraction, output_sino_with_bg, output_background
    Test: add_background(input_sinogram, randoms_fraction=param_randoms_fraction)
    Note: add_background uses an rng but for the default case with no rng
          the background is deterministic (constant uniform level).
    """
    N = 64
    phantom = create_activity_phantom(N)
    theta = np.linspace(0, 180, 30, endpoint=False)
    sinogram = pet_forward_project(phantom, theta)
    randoms_fraction = 0.1

    sino_with_bg, background = add_background(sinogram, randoms_fraction)

    path = os.path.join(FIXTURE_DIR, "physics_model_background.npz")
    np.savez(path,
             input_sinogram=sinogram,
             param_randoms_fraction=np.float64(randoms_fraction),
             output_sino_with_bg=sino_with_bg,
             output_background=background)
    print(f"  Saved {path}  bg_level={background[0,0]:.4f}")


def gen_preprocessing_preprocess():
    """Fixture: preprocessing_preprocess.npz

    Keys: input_sinogram, output_sinogram
    Test: preprocess_sinogram(input_sinogram) matches output_sinogram
    """
    # Create a sinogram with batch dim (1, n_radial, n_angles)
    # Include some negative values to test clipping
    rng = np.random.default_rng(42)
    sino_2d = rng.random((20, 30)) * 10 - 1  # some negatives
    input_sinogram = sino_2d[np.newaxis]  # (1, 20, 30)

    output_sinogram = preprocess_sinogram(input_sinogram)

    path = os.path.join(FIXTURE_DIR, "preprocessing_preprocess.npz")
    np.savez(path,
             input_sinogram=input_sinogram,
             output_sinogram=output_sinogram)
    print(f"  Saved {path}  input={input_sinogram.shape} output={output_sinogram.shape}")


if __name__ == "__main__":
    print("Generating pet_mlem fixtures...")
    gen_generate_data_phantom()
    gen_physics_model_forward()
    gen_physics_model_backproject()
    gen_physics_model_sensitivity()
    gen_physics_model_background()
    gen_preprocessing_preprocess()
    print("Done!")
