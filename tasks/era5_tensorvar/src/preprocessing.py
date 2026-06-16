"""I/O for the ERA5 Tensor-Var task.

Loads the bundled single-sample observation, ground truth, metadata and
latitude weight matrix, and resolves the four pretrained checkpoints
(`forward_model.pt`, `C_forward.pt`, `inverse_model.pt`, `z_b.pt`). The
checkpoints are *not* shipped with the benchmark — `download_pretrained_weights`
fetches them from the upstream Tensor-Var release on Google Drive on first use.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .physics_model import ERA5ForwardModel, ERA5InverseModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRETRAINED_GDRIVE_ID = "1Rmj18sLiHk6OmOMOoHKHS6Gpi_ZRKhlY"
PRETRAINED_FILENAMES = (
    "forward_model.pt",
    "C_forward.pt",
    "inverse_model.pt",
    "z_b.pt",
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_observation(data_dir: str | os.PathLike = "data") -> dict:
    """Return every key from `data/raw_data.npz` as a numpy array."""
    path = Path(data_dir) / "raw_data.npz"
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def load_ground_truth(data_dir: str | os.PathLike = "data") -> dict:
    """Return every key from `data/ground_truth.npz` as a numpy array."""
    path = Path(data_dir) / "ground_truth.npz"
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def load_metadata(data_dir: str | os.PathLike = "data") -> dict:
    """Return parsed `data/meta_data.json`."""
    with (Path(data_dir) / "meta_data.json").open("r") as fh:
        return json.load(fh)


def select_sample(arrays: dict, sample_index: int = 0) -> dict:
    """Drop the leading batch dimension by indexing into it."""
    return {k: v[sample_index] for k, v in arrays.items()}


# ---------------------------------------------------------------------------
# Pretrained checkpoint download + load
# ---------------------------------------------------------------------------
def _have_all_weights(weights_dir: Path) -> bool:
    return all((weights_dir / name).exists() for name in PRETRAINED_FILENAMES)


def download_pretrained_weights(weights_dir: str | os.PathLike) -> Path:
    """Ensure the four upstream Tensor-Var checkpoints exist in `weights_dir`.

    Downloads `ERA5_model_weights.zip` from the upstream Google Drive release
    via `gdown` if any of the expected files are missing, then extracts and
    moves them into place. Returns the absolute path to `weights_dir`.
    """
    weights_dir = Path(weights_dir).resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    if _have_all_weights(weights_dir):
        return weights_dir

    try:
        import gdown  # local import so the rest of the module works without it
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Downloading pretrained weights requires the `gdown` package. "
            "Install it with `pip install gdown` and re-run."
        ) from e

    zip_path = weights_dir / "ERA5_model_weights.zip"
    print(f"[INFO] Downloading Tensor-Var ERA5 checkpoints (~30 MB) to {zip_path}")
    gdown.download(id=PRETRAINED_GDRIVE_ID, output=str(zip_path), quiet=False)

    print(f"[INFO] Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(weights_dir)

    # The zip wraps the files in a `model_weights/` directory; flatten.
    nested = weights_dir / "model_weights"
    if nested.exists():
        for name in PRETRAINED_FILENAMES:
            src = nested / name
            if src.exists():
                src.replace(weights_dir / name)
        try:
            nested.rmdir()
        except OSError:
            pass

    if not _have_all_weights(weights_dir):
        missing = [n for n in PRETRAINED_FILENAMES if not (weights_dir / n).exists()]
        raise RuntimeError(
            f"Pretrained download finished but expected files are missing: {missing}"
        )

    zip_path.unlink(missing_ok=True)
    return weights_dir


def load_pretrained_models(
    weights_dir: str | os.PathLike,
    device: str | torch.device = "cpu",
) -> tuple[ERA5ForwardModel, ERA5InverseModel, torch.Tensor]:
    """Instantiate the forward and inverse models from on-disk checkpoints.

    Returns
    -------
    forward_model : ERA5ForwardModel
        Encoder + decoder + linear feature-space forward operator
        (`C_forward` is assigned as a tensor attribute).
    inverse_model : ERA5InverseModel
        Transformer-based inverse-observation network.
    z_b : torch.Tensor
        Background mean of the encoder feature vector, shape (512,).
    """
    weights_dir = Path(weights_dir)
    if not _have_all_weights(weights_dir):
        download_pretrained_weights(weights_dir)

    forward_model = ERA5ForwardModel()
    forward_model.load_state_dict(
        torch.load(weights_dir / "forward_model.pt", map_location="cpu", weights_only=False)
    )
    forward_model.C_forward = torch.load(
        weights_dir / "C_forward.pt", map_location=device, weights_only=False
    )
    forward_model.to(device)

    inverse_model = ERA5InverseModel()
    inverse_model.load_state_dict(
        torch.load(weights_dir / "inverse_model.pt", map_location="cpu", weights_only=False)
    )
    inverse_model.to(device)

    z_b = torch.load(weights_dir / "z_b.pt", map_location="cpu", weights_only=False)

    return forward_model, inverse_model, z_b


def default_covariances(hidden_dim: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the (B, R, Q) covariance matrices used by Tensor-Var when
    no calibrated covariances are shipped (the released checkpoint zip does
    not include them, so the upstream notebook falls back to identities)."""
    B = np.eye(hidden_dim) * 0.1
    R = np.eye(hidden_dim)
    Q = np.eye(hidden_dim) * 0.1
    return B, R, Q
