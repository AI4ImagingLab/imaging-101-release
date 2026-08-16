#!/usr/bin/env python
"""Validate FPM output archives against the run-local contract.
Usage: python scripts/script_001.py <case_id> <No> [outputs_dir]
Reads only run-local outputs; no private/reference data.
"""
import sys
import numpy as np


def main():
    if len(sys.argv) < 3:
        print("usage: validate.py <case_id> <No> [outputs_dir]")
        return 2
    case_id = sys.argv[1]
    No = int(sys.argv[2])
    outputs_dir = sys.argv[3] if len(sys.argv) > 3 else "outputs"
    path = f"{outputs_dir}/{case_id}.npz"
    try:
        arr = np.load(path)
    except Exception as e:
        print(f"FAIL: cannot load {path}: {e}")
        return 1
    if "object" not in arr.files:
        print(f"FAIL: key 'object' missing in {path}; keys={arr.files}")
        return 1
    obj = arr["object"]
    ok = True
    if obj.shape != (No, No):
        print(f"FAIL: shape {obj.shape} != {(No, No)}")
        ok = False
    if obj.dtype not in (np.complex64, np.complex128):
        print(f"FAIL: dtype {obj.dtype} not complex64/128")
        ok = False
    if not np.isfinite(obj).all():
        print("FAIL: non-finite values present")
        ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
