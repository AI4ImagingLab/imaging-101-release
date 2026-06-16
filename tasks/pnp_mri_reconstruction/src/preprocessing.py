import numpy as np
import json
import os


def load_observation(data_dir="data"):
    """Load MRI ground truth image from raw_data.npz.

    Returns:
        dict with key 'img': ndarray (N, N) — raw MRI image
    """
    data = np.load(os.path.join(data_dir, "raw_data.npz"))
    img = data["img"]
    # Strip batch dimension if present
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    return {"img": img}


def load_metadata(data_dir="data"):
    """Load imaging parameters from meta_data JSON file.

    Returns:
        dict with keys: image_size, num_lines, num_iter, step_size,
        patch_size, stride, state_num, sigma, model_checkpoints
    """
    with open(os.path.join(data_dir, "meta_data.json"), "r") as f:
        metadata = json.load(f)
    return metadata


def normalize_image(img):
    """Normalize image to [0, 1] range.

    Args:
        img: ndarray — raw image

    Returns:
        ndarray — normalized image in [0, 1]
    """
    return (img - img.min()) / (img.max() - img.min())


def prepare_data(data_dir="data"):
    """Full data preparation pipeline.

    Loads the MRI image, normalizes it, generates the radial sampling mask,
    and computes the subsampled k-space measurements.

    Returns:
        tuple: (img, mask, y, metadata)
            img: ndarray (N, N) — normalized ground truth in [0, 1]
            mask: ndarray (N, N) — boolean radial sampling mask
            y: ndarray (N, N) — subsampled k-space measurements
            metadata: dict — imaging parameters
    """
    from src.physics_model import MRIForwardModel

    obs = load_observation(data_dir)
    metadata = load_metadata(data_dir)

    img = normalize_image(obs["img"])
    image_size = np.array(img.shape)

    mask = MRIForwardModel.generate_mask(image_size, metadata["num_lines"])
    model = MRIForwardModel(mask)
    y = model.forward(img)

    return img, mask, y, metadata
