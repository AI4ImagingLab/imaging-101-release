"""
Data generation: create simulated laminographic projections from a phantom volume.

This module was used to generate data/raw_data.npz from data/ground_truth.npz.
"""

import os
import json
import numpy as np

from src.physics_model import forward_project


def generate_projections(volume, n_angles, tilt, theta_range=(0.0, np.pi)):
    """Generate simulated laminographic projections from a 3D volume.

    Parameters
    ----------
    volume : numpy.ndarray, (nz, n, n) complex64
        The ground truth 3D complex refractive index volume.
    n_angles : int
        Number of projection angles.
    tilt : float
        Tilt angle in radians. pi/2 for standard tomography.
    theta_range : tuple of float
        (start, stop) range for uniformly spaced angles in radians.

    Returns
    -------
    projections : numpy.ndarray, (n_angles, n, n) complex64
        Simulated projection data.
    theta : numpy.ndarray, (n_angles,) float32
        Projection angles in radians.
    """
    theta = np.linspace(
        theta_range[0], theta_range[1],
        n_angles, endpoint=False,
        dtype=np.float32,
    )

    projections = forward_project(
        obj=volume,
        theta=theta,
        tilt=tilt,
    )

    return projections, theta


def generate_and_save(task_dir):
    """Load ground truth, simulate projections, and save as raw_data.npz.

    Parameters
    ----------
    task_dir : str
        Root directory of the task (contains data/ subfolder).
    """
    # Load ground truth
    gt_path = os.path.join(task_dir, 'data', 'ground_truth.npz')
    gt_data = np.load(gt_path)
    volume = gt_data['volume'][0]  # Remove batch dim: (128, 128, 128)

    # Load metadata
    meta_path = os.path.join(task_dir, 'data', 'meta_data.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    n_angles = meta['n_angles']
    tilt = float(meta['tilt_rad'])
    theta_range = tuple(meta['theta_range_rad'])

    # Generate projections
    print(f"Generating {n_angles} projections at tilt={np.degrees(tilt):.1f} deg...")
    projections, theta = generate_projections(
        volume=volume,
        n_angles=n_angles,
        tilt=tilt,
        theta_range=theta_range,
    )

    # Save with batch dimension
    out_path = os.path.join(task_dir, 'data', 'raw_data.npz')
    np.savez(
        out_path,
        projections=projections[np.newaxis],  # (1, n_angles, n, n)
        theta=theta[np.newaxis],              # (1, n_angles)
    )
    print(f"Saved raw_data.npz to {out_path}")
    print(f"  projections shape: {projections[np.newaxis].shape}")
    print(f"  theta shape: {theta[np.newaxis].shape}")


if __name__ == '__main__':
    task_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_and_save(task_dir)
