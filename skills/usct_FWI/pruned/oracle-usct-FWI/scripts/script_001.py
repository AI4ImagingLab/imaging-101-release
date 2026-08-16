#!/usr/bin/env python
"""Validate a USCT FWI reconstruction npz against the task contract.

Usage: python script_001.py outputs/<case_id>.npz
Exit code 0 = OK, 1 = violation.
"""
import sys
import numpy as np


def validate(path):
    errors = []
    warnings = []
    try:
        data = np.load(path)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: cannot load {path}: {e}")
        return 1
    if "vp_reconstructed" not in data.files:
        errors.append("missing key 'vp_reconstructed'")
        print("; ".join(errors))
        return 1
    arr = data["vp_reconstructed"]
    if tuple(arr.shape) != (480, 480):
        errors.append(f"shape {arr.shape} != (480, 480)")
    if arr.dtype not in (np.float32, np.float64):
        errors.append(f"dtype {arr.dtype} not in float32/float64")
    if not np.all(np.isfinite(arr)):
        errors.append("array contains non-finite values")
    if arr.size and (arr.min() < 1200 or arr.max() > 1800):
        warnings.append(
            f"velocity range [{float(arr.min()):.1f}, {float(arr.max()):.1f}] "
            "outside expected ~[1300, 1700]"
        )
    for w in warnings:
        print(f"WARN: {w}")
    if errors:
        print("FAIL: " + "; ".join(errors))
        return 1
    print(f"OK: {path} shape={arr.shape} dtype={arr.dtype}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python script_001.py <path.npz>")
        sys.exit(2)
    sys.exit(validate(sys.argv[1]))
