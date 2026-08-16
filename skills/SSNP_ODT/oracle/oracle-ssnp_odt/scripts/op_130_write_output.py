"""Validate and write the SSNP-ODT delta_n output to outputs/{case_id}.npz.

Self-contained; run from the eval run root. No external/source paths.
"""
import os
import sys
import numpy as np

EXPECTED_SHAPE = (1, 308, 256, 256)


def write_delta_n(case_id, dn_recon, out_dir="outputs"):
    arr = np.asarray(dn_recon)
    # accept (Nz,Ny,Nx) or (1,Nz,Ny,Nx)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.shape != EXPECTED_SHAPE:
        raise ValueError(f"delta_n shape {arr.shape} != {EXPECTED_SHAPE}")
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError("delta_n contains non-finite values")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{case_id}.npz")
    np.savez(path, delta_n=arr)
    return path


def validate_file(path):
    d = np.load(path)
    assert "delta_n" in d.files, "missing key delta_n"
    a = d["delta_n"]
    assert a.shape == EXPECTED_SHAPE, f"bad shape {a.shape}"
    assert a.dtype in (np.float32, np.float64), f"bad dtype {a.dtype}"
    assert np.all(np.isfinite(a)), "non-finite values"
    print(f"OK {path} shape={a.shape} dtype={a.dtype}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate_file(sys.argv[1])
    else:
        print("usage: python op_130_write_output.py outputs/case_000.npz")
