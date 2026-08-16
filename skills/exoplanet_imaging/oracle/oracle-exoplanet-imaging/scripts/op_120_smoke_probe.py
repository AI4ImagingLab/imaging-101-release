#!/usr/bin/env python
"""Smoke-check written submissions against the output schema.

Usage: python scripts/op_120_smoke_probe.py [outputs_dir]
"""
import glob
import os
import sys
import numpy as np


def check(path):
    d = np.load(path)
    assert "klip_image" in d.files, f"{path}: missing key 'klip_image'"
    a = d["klip_image"]
    assert a.shape == (100, 100), f"{path}: shape {a.shape} != (100,100)"
    assert a.dtype in (np.float32, np.float64), f"{path}: dtype {a.dtype}"
    assert np.isfinite(a).all(), f"{path}: non-finite values present"
    return True


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    files = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    if not files:
        print(f"no npz files under {out_dir}")
        return 1
    for f in files:
        check(f)
        print(f"OK {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
