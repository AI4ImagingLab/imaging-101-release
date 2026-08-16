#!/usr/bin/env python
"""Validate fan-beam CT submission files against the output contract.

Usage: python scripts/op_110_validate_submission.py outputs/case_000.npz
"""
import sys
import numpy as np


def validate(path):
    d = np.load(path)
    assert 'phantom' in d.files, f"missing key 'phantom' in {path}"
    a = d['phantom']
    assert a.shape == (128, 128), f"bad shape {a.shape}, expected (128, 128)"
    assert a.dtype in (np.float32, np.float64), f"bad dtype {a.dtype}"
    assert np.all(np.isfinite(a)), "non-finite values present"
    print(f"OK: {path} shape={a.shape} dtype={a.dtype}")


if __name__ == '__main__':
    paths = sys.argv[1:] or ['outputs/case_000.npz']
    for p in paths:
        validate(p)
