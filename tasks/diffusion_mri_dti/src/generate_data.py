"""
Generate synthetic diffusion MRI data for DTI fitting.

Creates a 2D slice phantom with spatially varying diffusion tensors
assigned per tissue region. Each region has characteristic FA, MD, and
principal diffusion direction.
"""

import os
import json
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from src.physics_model import (
    tensor_from_eig,
    elements_from_tensor,
    stejskal_tanner_signal,
    add_rician_noise,
    compute_fa,
    compute_md,
)


def generate_gradient_table(n_directions=30, b_value=1000.0, n_b0=1, seed=42):
    """Generate a gradient table with uniformly distributed directions.

    Uses the electrostatic repulsion approach approximated by a Fibonacci
    sphere to distribute gradient directions uniformly on the hemisphere.

    Parameters
    ----------
    n_directions : int
        Number of diffusion-weighted gradient directions.
    b_value : float
        b-value for DWI volumes, in s/mm^2.
    n_b0 : int
        Number of b=0 volumes (placed at the beginning).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    bvals : np.ndarray
        b-values, shape (n_b0 + n_directions,).
    bvecs : np.ndarray
        Gradient directions, shape (n_b0 + n_directions, 3).
    """
    rng = np.random.default_rng(seed)

    # Fibonacci sphere for uniform distribution
    indices = np.arange(n_directions, dtype=np.float64)
    golden_ratio = (1 + np.sqrt(5)) / 2

    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_directions)

    gx = np.sin(phi) * np.cos(theta)
    gy = np.sin(phi) * np.sin(theta)
    gz = np.cos(phi)

    dwi_bvecs = np.column_stack([gx, gy, gz])
    # Normalize
    norms = np.linalg.norm(dwi_bvecs, axis=1, keepdims=True)
    dwi_bvecs = dwi_bvecs / norms

    # Combine b0 + DWI
    b0_bvecs = np.zeros((n_b0, 3), dtype=np.float64)
    bvecs = np.vstack([b0_bvecs, dwi_bvecs])

    bvals = np.zeros(n_b0 + n_directions, dtype=np.float64)
    bvals[n_b0:] = b_value

    return bvals, bvecs


def _rotation_matrix_from_axis(axis):
    """Create rotation matrix that maps [0,0,1] to the given unit vector."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(axis, z):
        return np.eye(3)
    if np.allclose(axis, -z):
        return np.diag([1.0, -1.0, -1.0])

    v = np.cross(z, axis)
    s = np.linalg.norm(v)
    c = np.dot(z, axis)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
    return R


# Tissue definitions: (intensity_low, intensity_high, eigenvalues, principal_direction, S0, label)
# eigenvalues in mm^2/s, sorted descending (lambda1 >= lambda2 >= lambda3)
TISSUE_REGIONS = [
    # Bone/scalp: low diffusivity, nearly isotropic
    (0.03, 0.15,
     [0.4e-3, 0.35e-3, 0.3e-3], [1, 0, 0], 0.4, "bone/scalp"),
    # White matter: moderate MD, high FA, horizontal fibers
    (0.15, 0.35,
     [1.2e-3, 0.3e-3, 0.3e-3], [1, 0, 0], 0.7, "white matter (horizontal)"),
    # Gray matter: moderate MD, low FA
    (0.35, 0.65,
     [0.9e-3, 0.8e-3, 0.7e-3], [0, 0, 1], 0.8, "gray matter"),
    # Deep gray matter: slightly higher FA with vertical direction
    (0.65, 0.85,
     [1.0e-3, 0.5e-3, 0.4e-3], [0, 1, 0], 0.9, "deep gray / white matter (vertical)"),
    # CSF: high MD, isotropic
    (0.85, 1.05,
     [3.0e-3, 3.0e-3, 3.0e-3], [0, 0, 1], 1.0, "CSF"),
]


def create_dti_phantom(N=128):
    """Create a 2D phantom with spatially varying diffusion tensors.

    Parameters
    ----------
    N : int
        Image size (N x N).

    Returns
    -------
    tensor_field : np.ndarray
        Diffusion tensor field, shape (N, N, 3, 3).
    S0_map : np.ndarray
        Proton density map, shape (N, N).
    tissue_mask : np.ndarray
        Boolean mask, shape (N, N).
    """
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (N, N), anti_aliasing=True, preserve_range=True)

    tensor_field = np.zeros((N, N, 3, 3), dtype=np.float64)
    S0_map = np.zeros((N, N), dtype=np.float64)

    for lo, hi, evals, direction, s0, label in TISSUE_REGIONS:
        region = (phantom >= lo) & (phantom < hi)
        evals = np.array(evals, dtype=np.float64)
        direction = np.array(direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)

        # Build rotation matrix that aligns principal direction
        R = _rotation_matrix_from_axis(direction)
        evecs = R  # columns are eigenvectors

        # Construct tensor: D = R @ diag(evals) @ R^T
        D = evecs @ np.diag(evals) @ evecs.T

        tensor_field[region] = D
        S0_map[region] = s0

    tissue_mask = S0_map > 0
    return tensor_field, S0_map, tissue_mask


def generate_synthetic_data(N=128, n_directions=30, b_value=1000.0,
                            sigma=0.02, seed=42):
    """Generate complete synthetic diffusion MRI dataset.

    Parameters
    ----------
    N : int
        Image size.
    n_directions : int
        Number of DWI gradient directions.
    b_value : float
        b-value in s/mm^2.
    sigma : float
        Rician noise level.
    seed : int
        Random seed.

    Returns
    -------
    data : dict
        All arrays needed for raw_data.npz, ground_truth.npz, and meta_data.json.
    """
    rng = np.random.default_rng(seed)

    # Generate gradient table
    bvals, bvecs = generate_gradient_table(
        n_directions=n_directions, b_value=b_value, n_b0=1, seed=seed,
    )

    # Create phantom
    tensor_field, S0_map, tissue_mask = create_dti_phantom(N)

    # Simulate DWI signal
    N_volumes = len(bvals)
    dwi_signal = np.zeros((N, N, N_volumes), dtype=np.float64)

    ys, xs = np.where(tissue_mask)
    for idx in range(len(ys)):
        y, x = ys[idx], xs[idx]
        D = tensor_field[y, x]
        s0 = S0_map[y, x]
        dwi_signal[y, x] = stejskal_tanner_signal(s0, D, bvals, bvecs)

    # Add Rician noise
    if sigma > 0:
        dwi_signal = add_rician_noise(dwi_signal, sigma, rng=rng)

    # Compute ground truth FA and MD from tensor field
    gt_eigenvalues = np.zeros((N, N, 3), dtype=np.float64)
    for idx in range(len(ys)):
        y, x = ys[idx], xs[idx]
        evals, _ = np.linalg.eigh(tensor_field[y, x])
        gt_eigenvalues[y, x] = np.sort(evals)[::-1]

    fa_map = compute_fa(gt_eigenvalues)
    fa_map = np.where(tissue_mask, fa_map, 0.0)
    md_map = compute_md(gt_eigenvalues)
    md_map = np.where(tissue_mask, md_map, 0.0)

    # Extract tensor elements for ground truth
    tensor_elements = elements_from_tensor(tensor_field)  # (N, N, 6)

    return {
        'dwi_signal': dwi_signal[np.newaxis, ...],           # (1, N, N, N_volumes)
        'bvals': bvals,                                       # (N_volumes,)
        'bvecs': bvecs,                                       # (N_volumes, 3)
        'fa_map': fa_map[np.newaxis, ...],                    # (1, N, N)
        'md_map': md_map[np.newaxis, ...],                    # (1, N, N)
        'tensor_elements': tensor_elements[np.newaxis, ...],  # (1, N, N, 6)
        'tissue_mask': tissue_mask[np.newaxis, ...],          # (1, N, N)
        'S0_map': S0_map[np.newaxis, ...],                    # (1, N, N)
        'N': N,
        'n_directions': n_directions,
        'b_value': b_value,
        'sigma': sigma,
    }


def save_data(data, task_dir):
    """Save generated data to task data/ directory.

    Parameters
    ----------
    data : dict
        Output from generate_synthetic_data.
    task_dir : str
        Path to the task root directory.
    """
    data_dir = os.path.join(task_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # raw_data.npz: DWI measurements + acquisition parameters
    np.savez_compressed(
        os.path.join(data_dir, 'raw_data.npz'),
        dwi_signal=data['dwi_signal'].astype(np.float32),
        bvals=data['bvals'].astype(np.float32),
        bvecs=data['bvecs'].astype(np.float32),
    )

    # ground_truth.npz: true tensor-derived maps
    np.savez_compressed(
        os.path.join(data_dir, 'ground_truth.npz'),
        fa_map=data['fa_map'].astype(np.float32),
        md_map=data['md_map'].astype(np.float32),
        tensor_elements=data['tensor_elements'].astype(np.float32),
        tissue_mask=data['tissue_mask'],
    )

    # meta_data.json
    meta = {
        "image_size": int(data['N']),
        "n_directions": int(data['n_directions']),
        "n_b0": 1,
        "b_value_s_per_mm2": float(data['b_value']),
        "noise_sigma": float(data['sigma']),
        "signal_model": "stejskal_tanner",
        "modality": "diffusion-weighted MRI",
        "field_of_view_mm": 220.0,
        "pixel_size_mm": 220.0 / int(data['N']),
    }
    with open(os.path.join(data_dir, 'meta_data.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    task_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = generate_synthetic_data()
    save_data(data, task_dir)
    print(f"Data saved to {os.path.join(task_dir, 'data')}")
    print(f"  dwi_signal: {data['dwi_signal'].shape}")
    print(f"  bvals: {data['bvals'].shape}")
    print(f"  fa_map: {data['fa_map'].shape}")
    print(f"  md_map: {data['md_map'].shape}")
