import os, json, numpy as np

def load_observation(data_dir="data"):
    """Load raw EIT observation data."""
    data = np.load(os.path.join(data_dir, "raw_data.npz"), allow_pickle=True)
    return dict(data)

def load_metadata(data_dir="data"):
    """Load experiment metadata."""
    with open(os.path.join(data_dir, "meta_data.json"), "r") as f:
        return json.load(f)

def prepare_data(data_dir="data"):
    """Load observation data and metadata."""
    obs = load_observation(data_dir)
    meta = load_metadata(data_dir)
    return obs, meta
