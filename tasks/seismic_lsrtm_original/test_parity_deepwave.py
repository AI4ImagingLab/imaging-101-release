"""Parity test: scratch C-PML Born solver against deepwave.scalar_born.

Both use 4th-order FD accuracy for a fair comparison.
"""

import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from src.physics_model import (
    forward_model as forward_scratch,
    born_forward_model as born_scratch,
    make_acquisition_geometry as make_geom_scratch,
    make_ricker_wavelet as make_wavelet_scratch,
)

import deepwave
from deepwave import scalar, scalar_born


def rel_l2(a, b):
    return float(torch.norm(a - b) / (torch.norm(b) + 1e-30))


def test_forward_homogeneous(device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ny, nx = 80, 50
    v = torch.ones(ny, nx, device=device) * 2000.0
    dx, dt, freq, nt = 4.0, 0.004, 25.0, 100
    n_shots = 2

    sl, rl = make_geom_scratch(n_shots=n_shots, d_source=20, first_source=5,
        n_receivers=10, d_receiver=4, device=device)
    w = make_wavelet_scratch(freq, nt, dt, n_shots, device)

    # deepwave with accuracy=4 to match scratch
    with torch.no_grad():
        rec_s = forward_scratch(v, dx, dt, w, sl, rl, freq)
        out_d = scalar(v, dx, dt, source_amplitudes=w, source_locations=sl,
            receiver_locations=rl, accuracy=4, pml_freq=freq)
        rec_d = out_d[-1]

    err = rel_l2(rec_s, rec_d)
    print(f"[Forward homogeneous] rel_l2 = {err:.2e}")
    return err


def test_born_homogeneous(device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ny, nx = 80, 50
    v = torch.ones(ny, nx, device=device) * 2000.0
    scatter = torch.randn(ny, nx, device=device) * 50.0
    dx, dt, freq, nt = 4.0, 0.004, 25.0, 100
    n_shots = 2

    sl, rl = make_geom_scratch(n_shots=n_shots, d_source=20, first_source=5,
        n_receivers=10, d_receiver=4, device=device)
    w = make_wavelet_scratch(freq, nt, dt, n_shots, device)

    with torch.no_grad():
        rec_s = born_scratch(v, scatter, dx, dt, w, sl, rl, freq)
        out_d = scalar_born(v, scatter, dx, dt, source_amplitudes=w,
            source_locations=sl, receiver_locations=rl, accuracy=4, pml_freq=freq)
        rec_d = out_d[-1]

    err = rel_l2(rec_s, rec_d)
    print(f"[Born homogeneous] rel_l2 = {err:.2e}")
    print(f"  scratch range: [{rec_s.min():.4e}, {rec_s.max():.4e}]")
    print(f"  deepwave range: [{rec_d.min():.4e}, {rec_d.max():.4e}]")
    return err


def test_born_marmousi(device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    raw = np.load("data/raw_data.npz")
    gt = np.load("data/ground_truth.npz")
    v_mig = torch.from_numpy(raw["v_mig"]).to(device)
    dx, dt, freq = float(raw["dx"]), float(raw["dt"]), float(raw["freq"])
    n_shots, n_rec, nt = int(raw["n_shots"]), int(raw["n_receivers"]), int(raw["nt"])

    sl, rl = make_geom_scratch(device=device)
    w = make_wavelet_scratch(freq, nt, dt, n_shots, device)
    scatter = torch.randn_like(v_mig) * 100

    print(f"[Born Marmousi] v_mig: {v_mig.shape}, {n_shots} shots")
    with torch.no_grad():
        rec_s = born_scratch(v_mig, scatter, dx, dt, w, sl, rl, freq)
        out_d = scalar_born(v_mig, scatter, dx, dt, source_amplitudes=w,
            source_locations=sl, receiver_locations=rl, accuracy=4, pml_freq=freq)
        rec_d = out_d[-1]

    err = rel_l2(rec_s, rec_d)
    print(f"  rel_l2 = {err:.2e}")
    return err


if __name__ == "__main__":
    print("=" * 60)
    print("Parity test: scratch C-PML Born solver vs deepwave")
    print("(Both using accuracy=4 for fair comparison)")
    print("=" * 60)

    e1 = test_forward_homogeneous()
    e2 = test_born_homogeneous()
    e3 = test_born_marmousi()

    print(f"\nSummary:")
    print(f"  Forward homo:  {e1:.2e}")
    print(f"  Born homo:     {e2:.2e}")
    print(f"  Born Marmousi: {e3:.2e}")

    if max(e1, e2, e3) < 1e-4:
        print("\nPASS: all within floating-point tolerance (< 1e-4)")
    else:
        print("\nFAIL: differences exceed tolerance")
