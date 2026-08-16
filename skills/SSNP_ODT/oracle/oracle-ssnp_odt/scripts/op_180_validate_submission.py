"""Validate an SSNP-ODT submission .npz against the output schema.

Usage:
    python op_180_validate_submission.py <path.npz> [Nz Ny Nx]
Defaults: Nz=308 Ny=256 Nx=256.
"""
import sys
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("usage: op_180_validate_submission.py <path.npz> [Nz Ny Nx]")
        sys.exit(2)
    path = sys.argv[1]
    nz, ny, nx = 308, 256, 256
    if len(sys.argv) >= 5:
        nz, ny, nx = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    data = np.load(path)
    assert "delta_n" in data.files, f"missing key 'delta_n'; found {data.files}"
    arr = data["delta_n"]
    exp = (1, nz, ny, nx)
    assert arr.shape == exp, f"shape {arr.shape} != expected {exp}"
    assert arr.dtype in (np.float32, np.float64), f"bad dtype {arr.dtype}"
    assert np.isfinite(arr).all(), "array contains non-finite values"
    print(f"OK: {path} delta_n shape={arr.shape} dtype={arr.dtype} "
          f"range=[{arr.min():.6g}, {arr.max():.6g}]")


if __name__ == "__main__":
    main()
