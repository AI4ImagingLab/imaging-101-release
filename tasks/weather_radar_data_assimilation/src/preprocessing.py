"""Data loading and normalization for weather radar data assimilation."""

import os
import json
import numpy as np


def load_raw_data(data_dir: str) -> dict:
    """Load raw_data.npz and return dict with condition_frames, observations, observation_mask."""
    path = os.path.join(data_dir, "raw_data.npz")
    data = np.load(path)
    return {
        "condition_frames": data["condition_frames"],  # (1, 6, 128, 128)
        "observations": data["observations"],          # (1, 3, 128, 128)
        "observation_mask": data["observation_mask"],   # (1, 1, 128, 128)
    }


def load_ground_truth(data_dir: str) -> np.ndarray:
    """Load ground_truth.npz and return target_frames array."""
    path = os.path.join(data_dir, "ground_truth.npz")
    data = np.load(path)
    return data["target_frames"]  # (1, 3, 128, 128)


def load_meta_data(data_dir: str) -> dict:
    """Load meta_data.json."""
    path = os.path.join(data_dir, "meta_data.json")
    with open(path) as f:
        return json.load(f)


def scale_to_model(x: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0,1] to model space: (x - 0.5) * 10."""
    return (x - 0.5) * 10.0


def scale_from_model(x: np.ndarray) -> np.ndarray:
    """Scale from model space back to [0,1]: x / 10 + 0.5."""
    return x / 10.0 + 0.5
