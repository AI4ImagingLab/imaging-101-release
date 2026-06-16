"""
Data Preprocessing for VarNet MRI Reconstruction
==================================================

Loads multi-coil k-space data and RSS ground truth from npz files.
Handles k-space undersampling via equispaced masks.

EquiSpacedMaskFunc ported from fastMRI (facebookresearch/fastMRI):
    fastmri/data/subsample.py
"""

import os
import json
import contextlib
import numpy as np
import torch
from typing import Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Equi-spaced mask generation (ported from fastmri/data/subsample.py)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _temp_seed(rng: np.random.RandomState, seed):
    """Temporarily set RNG seed, restoring state on exit.

    Ported from fastMRI subsample.py temp_seed().
    """
    if seed is None:
        yield
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class EquiSpacedMaskFunc:
    """Generate equi-spaced undersampling masks for MRI k-space.

    Produces a mask that samples every `acceleration`-th line,
    plus a contiguous block of center (ACS) lines.

    Ported from fastMRI subsample.py MaskFunc + EquiSpacedMaskFunc.

    Parameters
    ----------
    center_fractions : list of float
        Fraction of low-frequency columns to retain.
    accelerations : list of int
        Acceleration factors.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        seed: Optional[int] = None,
    ):
        if len(center_fractions) != len(accelerations):
            raise ValueError(
                "Number of center fractions should match number of accelerations."
            )
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Generate the undersampling mask.

        Parameters
        ----------
        shape : tuple
            Shape of k-space, at least 3 dims. Mask applies to dim -2.
        offset : int or None
            Starting offset for equispaced lines. Random if None.
        seed : int or None
            Seed for reproducibility.

        Returns
        -------
        mask : Tensor, broadcastable to shape
        num_low_frequencies : int
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with _temp_seed(self.rng, seed):
            num_cols = shape[-2]

            # Choose acceleration/center fraction pair
            choice = self.rng.randint(len(self.center_fractions))
            center_fraction = self.center_fractions[choice]
            acceleration = self.accelerations[choice]

            num_low_frequencies = round(num_cols * center_fraction)

            # Center mask (ACS lines)
            center_mask = np.zeros(num_cols, dtype=np.float32)
            pad = (num_cols - num_low_frequencies + 1) // 2
            center_mask[pad: pad + num_low_frequencies] = 1

            # Acceleration mask (equi-spaced)
            if offset is None:
                offset = self.rng.randint(0, high=round(acceleration))
            accel_mask = np.zeros(num_cols, dtype=np.float32)
            accel_mask[offset::acceleration] = 1

            # Combine: union of center and acceleration masks
            mask = np.maximum(center_mask, accel_mask)

            # Reshape to broadcast over other dims
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape))

        return mask, num_low_frequencies


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_observation(data_dir: str = "data") -> dict:
    """Load multi-coil k-space from raw_data.npz."""
    path = os.path.join(data_dir, "raw_data.npz")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_ground_truth(data_dir: str = "data") -> np.ndarray:
    """Load RSS ground truth. Returns (N, H, W) float32."""
    return np.load(os.path.join(data_dir, "ground_truth.npz"))["image"]


def load_metadata(data_dir: str = "data") -> dict:
    """Load imaging parameters."""
    with open(os.path.join(data_dir, "meta_data.json")) as f:
        return json.load(f)


def get_complex_kspace(obs_data: dict) -> np.ndarray:
    """Reconstruct complex k-space. Returns (N, Nc, H, W) complex64."""
    return (obs_data["kspace_real"] + 1j * obs_data["kspace_imag"]).astype(np.complex64)


def apply_mask(kspace_slice: np.ndarray, acceleration: int = 4,
               center_fraction: float = 0.08, seed: int = 42):
    """
    Apply equispaced undersampling mask to a single k-space slice.

    Parameters
    ----------
    kspace_slice : (Nc, H, W) complex64
    acceleration : int
    center_fraction : float
    seed : int

    Returns
    -------
    masked_kspace : Tensor, (Nc, H, W, 2) float32
    mask : Tensor, boolean
    """
    kspace_torch = torch.view_as_real(
        torch.from_numpy(kspace_slice.astype(np.complex64))
    )  # (Nc, H, W, 2)

    mask_func = EquiSpacedMaskFunc(
        center_fractions=[center_fraction],
        accelerations=[acceleration],
    )
    shape = (1,) * len(kspace_torch.shape[:-3]) + tuple(kspace_torch.shape[-3:])
    mask, _ = mask_func(shape, seed=seed)

    masked_kspace = kspace_torch * mask + 0.0
    return masked_kspace, mask.bool()


def prepare_data(data_dir: str = "data"):
    """
    Load all data needed for VarNet reconstruction.

    Returns
    -------
    kspace : (N, Nc, H, W) complex64
    ground_truth : (N, H, W) float32
    metadata : dict
    """
    obs = load_observation(data_dir)
    kspace = get_complex_kspace(obs)
    gt = load_ground_truth(data_dir)
    meta = load_metadata(data_dir)
    return kspace, gt, meta
