"""Generate and save synthetic seismic data from the Marmousi model.

Downloads the velocity model, preprocesses it, runs forward modeling using
our scratch wave propagation, and saves the resulting dataset.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_marmousi, preprocess_velocity, make_initial_model
from src.physics_model import (
    make_acquisition_geometry,
    make_ricker_wavelet,
    forward_model,
)


def generate_and_save(
    vp_bin_path: str,
    output_npz: str,
    device: torch.device = torch.device("cpu"),
    n_shots: int = 10,
    n_receivers: int = 93,
    nt: int = 1350,
    dt: float = 4e-3,
    freq: float = 5.0,
) -> None:
    """Load Marmousi, simulate shot gathers, save dataset.

    Args:
        vp_bin_path: Path to Marmousi vp.bin (6.6 MB, float32).
        output_npz: Output .npz path.
        device: PyTorch device.
        n_shots: Number of shots. Default 10.
        n_receivers: Receivers per shot. Default 93.
        nt: Number of time samples. Default 1350.
        dt: Time step in seconds. Default 4e-3.
        freq: Source frequency. Default 5.0 Hz.
    """
    print("Loading Marmousi velocity model...")
    v_full = load_marmousi(vp_bin_path)
    v_true, dx = preprocess_velocity(v_full, factor=5)
    v_init = make_initial_model(v_true, sigma=25.0)
    spacing = (dx, dx)
    ny, nx = v_true.shape
    print(f"  Model shape: {ny} x {nx}, dx={dx} m")

    print("Setting up acquisition geometry...")
    source_loc, receiver_loc = make_acquisition_geometry(
        ny, n_shots=n_shots, n_receivers=n_receivers, device=device
        # ny is the horizontal dimension (first axis)
    )
    source_amp = make_ricker_wavelet(freq, nt, dt, n_shots, device=device)

    print("Running forward simulation on true model (this may take a while)...")
    v_true_dev = v_true.to(device)
    with torch.no_grad():
        observed_data = forward_model(
            v_true_dev, spacing, dt, source_amp, source_loc, receiver_loc, freq
        )
    observed_data = observed_data.cpu()
    print(f"  observed_data shape: {observed_data.shape}")

    print(f"Saving dataset to {output_npz} ...")
    np.savez(
        output_npz,
        v_true=v_true.numpy(),
        v_init=v_init.numpy(),
        observed_data=observed_data.numpy(),
        dx=np.float32(dx),
        n_shots=np.int32(n_shots),
        n_receivers=np.int32(n_receivers),
        nt=np.int32(nt),
        dt=np.float32(dt),
        freq=np.float32(freq),
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate seismic dataset from Marmousi.")
    parser.add_argument("vp_bin", help="Path to Marmousi vp.bin")
    parser.add_argument("--output", default="data/raw_data.npz")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    generate_and_save(args.vp_bin, args.output, torch.device(args.device))
