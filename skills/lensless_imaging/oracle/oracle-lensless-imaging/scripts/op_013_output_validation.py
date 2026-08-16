#!/usr/bin/env python
"""Validate a lensless reconstruction npz against the submission schema.

Usage:
    python scripts/op_013_output_validation.py outputs/<case_id>.npz

Exits non-zero and prints FAIL messages if any check fails.
"""
import sys
import numpy as np

EXPECTED_SHAPE = (760, 1014, 3)
ALLOWED_DTYPES = ("float32", "float64")


def main():
    if len(sys.argv) != 2:
        print("usage: python op_013_output_validation.py <npz>")
        sys.exit(2)
    path = sys.argv[1]
    ok = True
    try:
        z = np.load(path)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot load {path}: {e}")
        sys.exit(1)
    if "reconstruction" not in z.files:
        print(f"FAIL: missing key 'reconstruction'; keys={z.files}")
        sys.exit(1)
    arr = z["reconstruction"]
    if tuple(arr.shape) != EXPECTED_SHAPE:
        print(f"FAIL: shape {arr.shape} != {EXPECTED_SHAPE}")
        ok = False
    if str(arr.dtype) not in ALLOWED_DTYPES:
        print(f"FAIL: dtype {arr.dtype} not in {ALLOWED_DTYPES}")
        ok = False
    if not np.isfinite(arr).all():
        print("FAIL: array contains non-finite values")
        ok = False
    if ok:
        print(f"OK: {path} shape={arr.shape} dtype={arr.dtype} "
              f"min={float(arr.min()):.4f} max={float(arr.max()):.4f}")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
