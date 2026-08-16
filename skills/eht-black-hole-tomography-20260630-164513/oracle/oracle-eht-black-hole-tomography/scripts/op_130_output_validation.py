#!/usr/bin/env python3
"""Validate BH-NeRF emission outputs against the task contract.

Usage: python scripts/op_130_output_validation.py [outputs_dir]
Defaults to ./outputs. Exits non-zero on any failure.
"""
import sys
import glob
import os
import numpy as np


def check_file(path):
    errs = []
    try:
        data = np.load(path)
    except Exception as e:  # noqa: BLE001
        return [f"{path}: cannot load ({e})"]
    if 'emission_3d' not in getattr(data, 'files', []):
        return [f"{path}: missing key 'emission_3d'"]
    arr = data['emission_3d']
    if tuple(arr.shape) != (64, 64, 64):
        errs.append(f"{path}: shape {tuple(arr.shape)} != (64, 64, 64)")
    if arr.dtype not in (np.float32, np.float64):
        errs.append(f"{path}: dtype {arr.dtype} not float32/float64")
    if not np.isfinite(arr).all():
        errs.append(f"{path}: contains non-finite values")
    return errs


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'outputs'
    files = sorted(glob.glob(os.path.join(out_dir, '*.npz')))
    if not files:
        print(f"No .npz files found in {out_dir}")
        return 1
    all_errs = []
    for f in files:
        all_errs.extend(check_file(f))
    if all_errs:
        for e in all_errs:
            print("FAIL", e)
        return 1
    print(f"OK: {len(files)} output file(s) passed validation")
    return 0


if __name__ == '__main__':
    sys.exit(main())
