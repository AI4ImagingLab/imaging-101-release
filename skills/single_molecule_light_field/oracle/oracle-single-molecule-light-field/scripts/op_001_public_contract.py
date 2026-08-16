#!/usr/bin/env python
"""Validate a submission NPZ against the required output schema.

Usage: python op_001_public_contract.py outputs/case_000.npz
Checks: three required scalar keys, correct dtype families, finite values.
"""
import sys
import numpy as np

REQUIRED = {
    "n_locs_3d_filtered": ("i",),
    "median_lateral_err_nm": ("f",),
    "median_axial_err_nm": ("f",),
}


def main(path):
    d = np.load(path)
    ok = True
    for key, kinds in REQUIRED.items():
        if key not in d:
            print(f"MISSING key: {key}")
            ok = False
            continue
        arr = d[key]
        if arr.shape != ():
            print(f"BAD shape for {key}: {arr.shape} (want scalar [])")
            ok = False
        if arr.dtype.kind not in kinds:
            print(f"BAD dtype for {key}: {arr.dtype} (want kind {kinds})")
            ok = False
        if not np.isfinite(np.asarray(arr, dtype=float)).all():
            print(f"NON-FINITE value for {key}: {arr}")
            ok = False
        else:
            print(f"OK {key} = {arr} dtype={arr.dtype}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "outputs/case_000.npz"))
