#!/usr/bin/env python
"""Cheap smoke probe for the FPM-INR pipeline. Non-mutating.

Usage:
  public/envs/runtime/.venv-posix/bin/python \
      ${CLAUDE_SKILL_DIR}/scripts/op_130_smoke_probe.py public/cases/case_000/input_data
"""
import os
import sys
import numpy as np


def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "public/cases/case_000/input_data"
    ok = True
    try:
        import torch
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("torch", torch.__version__, "device", dev)
    except Exception as e:  # noqa: BLE001
        print("torch import FAILED:", e); sys.exit(1)

    raw_path = os.path.join(in_dir, "raw_data.npz")
    d = np.load(raw_path)
    il = d["I_low"]; na = d["na_calib"]
    print("I_low", il.shape, il.dtype, "na_calib", na.shape)
    if il.shape != (1, 1024, 1024, 68):
        print("WARN unexpected I_low shape"); ok = False
    if na.shape != (1, 68, 2):
        print("WARN unexpected na_calib shape"); ok = False

    # FFT roundtrip with the centered convention used by the forward model
    x = torch.randn(2, 16, 16) + 1j * torch.randn(2, 16, 16)
    X = torch.fft.fftshift(torch.fft.fft2(x))
    xr = torch.fft.ifft2(torch.fft.ifftshift(X))
    err = (x - xr).abs().max().item()
    print("fft roundtrip max abs err", err)
    if err > 1e-4:
        print("WARN fft roundtrip error high"); ok = False

    lam = 0.5126; mag = float(d["mag"]); px = float(d["dpix_c"]); NA = float(d["na_cal"])
    k0 = 2 * np.pi / lam; D_pixel = px / mag; kmax = NA * k0
    print("k0", k0, "D_pixel", D_pixel, "kmax", kmax)
    if not (np.isfinite([k0, D_pixel, kmax]).all() and min(k0, D_pixel, kmax) > 0):
        print("WARN optical scalars invalid"); ok = False

    print("PROBE", "OK" if ok else "HAD-WARNINGS")


if __name__ == "__main__":
    main()
