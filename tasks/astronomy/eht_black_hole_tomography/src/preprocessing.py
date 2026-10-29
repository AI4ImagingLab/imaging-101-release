"""
Data loading and preprocessing for BH-NeRF task.
"""

import json
import os
import numpy as np
import torch


def load_metadata(data_dir="data"):
    """
    Load meta_data JSON file.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    metadata : dict
        Dictionary with simulation and training parameters.
    """
    path = os.path.join(data_dir, "meta_data.json")
    with open(path, 'r') as f:
        return json.load(f)


def load_observation(data_dir="data"):
    """
    Load raw_data.npz and organize into structured dict of torch tensors.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    obs_data : dict with keys:
        'ray_coords' : torch.Tensor, shape (3, num_alpha, num_beta, ngeo)
        'Omega' : torch.Tensor
        'g_doppler' : torch.Tensor
        'dtau' : torch.Tensor
        'Sigma' : torch.Tensor
        't_geo' : torch.Tensor
        't_frames' : np.ndarray, shape (n_frames,)
        'images' : torch.Tensor, shape (n_frames, num_alpha, num_beta) — noisy
        'fov_M' : float
        't_start_obs' : float
        't_injection' : float
    """
    path = os.path.join(data_dir, "raw_data.npz")
    data = np.load(path)

    obs_data = {
        'ray_coords': torch.tensor(np.stack([
            data['ray_x'], data['ray_y'], data['ray_z']
        ]), dtype=torch.float32),
        'Omega': torch.tensor(data['Omega'], dtype=torch.float32),
        'g_doppler': torch.tensor(data['g_doppler'], dtype=torch.float32),
        'dtau': torch.tensor(data['ray_dtau'], dtype=torch.float32),
        'Sigma': torch.tensor(data['ray_Sigma'], dtype=torch.float32),
        't_geo': torch.tensor(data['ray_t_geo'], dtype=torch.float32),
        't_frames': data['t_frames'].astype(np.float32),
        'images': torch.tensor(data['images_noisy'], dtype=torch.float32),
        'fov_M': float(data['fov_M']),
        't_start_obs': float(data['t_start_obs']),
        't_injection': float(data['t_injection']),
    }
    return obs_data


def load_ground_truth(data_dir="data"):
    """
    Load ground-truth emission and images for evaluation.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    ground_truth : dict with keys:
        'emission_3d' : np.ndarray, shape (res, res, res)
        'images' : np.ndarray, shape (n_frames, num_alpha, num_beta)
        'rot_axis' : np.ndarray, shape (3,)
    """
    path = os.path.join(data_dir, "ground_truth.npz")
    data = np.load(path)

    return {
        'emission_3d': data['emission_3d'],
        'images': data['images'],
        'rot_axis': data['rot_axis'],
    }


def prepare_data(data_dir="data"):
    """
    Combined data loader.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    obs_data : dict
    ground_truth : dict
    metadata : dict
    """
    metadata = load_metadata(data_dir)
    obs_data = load_observation(data_dir)
    ground_truth = load_ground_truth(data_dir)
    return obs_data, ground_truth, metadata
