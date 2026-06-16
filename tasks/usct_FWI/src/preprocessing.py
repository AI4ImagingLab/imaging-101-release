"""Data loading, coordinate setup, restriction operator, and masking for USCT FWI."""

import json
import os
import numpy as np
import torch
from scipy.sparse import coo_matrix
from typing import Tuple, Dict, Any


def load_metadata(data_dir: str = "data") -> Dict[str, Any]:
    """Load meta_data.json."""
    with open(os.path.join(data_dir, "meta_data.json")) as f:
        return json.load(f)


def load_observations(data_dir: str = "data", freq: float = None) -> dict:
    """Load observation data from raw_data.npz.

    Args:
        data_dir: path to data directory
        freq: if specified, return only dobs for that frequency as torch.Tensor on CUDA.
              if None, return all dobs as a dict {freq_str: tensor}.

    Returns:
        dict with 'receiver_ix', 'receiver_iy', and dobs tensors
    """
    raw = np.load(os.path.join(data_dir, "raw_data.npz"))
    # Squeeze batch dimension (stored as (1, ...))
    result = {
        "receiver_ix": raw["receiver_ix"].squeeze(0).astype(np.float32),
        "receiver_iy": raw["receiver_iy"].squeeze(0).astype(np.float32),
    }
    if freq is not None:
        freq_str = f"{freq:g}"
        key = f"dobs_{freq_str}"
        result["dobs"] = torch.tensor(
            raw[key].squeeze(0), dtype=torch.complex64
        ).cuda()
    else:
        result["dobs_all"] = {}
        for k in raw.files:
            if k.startswith("dobs_"):
                freq_str = k[5:]
                result["dobs_all"][freq_str] = torch.tensor(
                    raw[k].squeeze(0), dtype=torch.complex64
                ).cuda()
    return result


def load_baseline_reference(data_dir: str = "data") -> np.ndarray:
    """Load baseline reference velocity model. Returns float32 (nx, ny)."""
    ref = np.load(os.path.join(data_dir, "baseline_reference.npz"))
    return ref["vp_reconstructed"].squeeze(0).astype(np.float32)


def build_restriction_operator(
    ix: np.ndarray, iy: np.ndarray, nx: int, ny: int
) -> torch.Tensor:
    """Build sparse restriction operator R: (n_rec, nx*ny).

    Maps flattened full wavefield to receiver locations.
    ix, iy are 1-indexed positions (from MATLAB convention).

    Returns:
        R: sparse CUDA tensor (n_rec, nx*ny)
    """
    n_rec = len(ix)
    # Convert to linear indices (0-indexed)
    src_linear_index = (iy - 1) * ny + ix
    icol = (src_linear_index - 1).astype(np.int32).flatten()
    irow = np.arange(n_rec, dtype=np.int32)
    vals = np.ones(n_rec, dtype=np.float32)

    R_coo = coo_matrix((vals, (irow, icol)), shape=(n_rec, ny * nx))
    indices = torch.LongTensor(np.vstack((R_coo.row, R_coo.col)))
    values = torch.FloatTensor(R_coo.data)
    R = torch.sparse_coo_tensor(indices, values, torch.Size(R_coo.shape))
    return R.cuda()


def create_dobs_masks(
    dobs: torch.Tensor,
    ix: np.ndarray,
    iy: np.ndarray,
    dh: float,
    mute_dist: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create masked observations and masks for source estimation and misfit.

    Args:
        dobs: complex64 (n_rec, n_rec) observation data
        ix, iy: float32 (n_rec,) 1-indexed receiver positions
        dh: grid spacing in um
        mute_dist: muting distance in um

    Returns:
        dobs_masked: complex64 (n_rec, n_rec)
        mask_esi: bool (n_rec, n_rec) for source estimation
        mask_misfit: bool (n_rec, n_rec) for misfit computation
    """
    n_rec = len(ix)
    mask = np.ones((n_rec, n_rec), dtype=bool)
    for isrc in range(n_rec):
        for irec in range(n_rec):
            x2 = (ix[isrc] - ix[irec]) ** 2
            y2 = (iy[isrc] - iy[irec]) ** 2
            if np.sqrt(x2 + y2) * dh < mute_dist:
                mask[irec, isrc] = False

    mask_esi = torch.from_numpy(mask).cuda()
    mask_misfit = mask_esi.clone()
    mask_misfit[abs(dobs) == 0] = False
    dobs_masked = dobs * mask_esi
    return dobs_masked, mask_esi, mask_misfit


def create_initial_slowness(
    nx: int, ny: int, v0: float, device: str = "cuda"
) -> torch.Tensor:
    """Create homogeneous initial slowness model 1/v0.

    Returns:
        slowness: float32 (nx, ny) on device
    """
    return (torch.ones(nx, ny, dtype=torch.float32) / v0).to(device)
