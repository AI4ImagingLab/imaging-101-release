"""Validate outputs/{case_id}.npz against the task contract.
Usage: python scripts/op_120_validate_submission.py outputs/case_000.npz
Run from the run root with the run-local python.
"""
import sys
import numpy as np

EXPECTED_SHAPE = (128, 128, 128)
ALLOWED_DTYPES = ('complex64', 'complex128')

def validate(path):
    d = np.load(path)
    assert 'volume' in d.files, f"missing 'volume' key in {path}"
    v = d['volume']
    assert tuple(v.shape) == EXPECTED_SHAPE, f"bad shape {v.shape}, want {EXPECTED_SHAPE}"
    assert str(v.dtype) in ALLOWED_DTYPES, f"bad dtype {v.dtype}, want one of {ALLOWED_DTYPES}"
    assert np.isfinite(v.real).all() and np.isfinite(v.imag).all(), 'non-finite values present'
    print(f"OK {path}: shape={v.shape} dtype={v.dtype}")

if __name__ == '__main__':
    paths = sys.argv[1:] or ['outputs/case_000.npz']
    for p in paths:
        validate(p)
""""""
