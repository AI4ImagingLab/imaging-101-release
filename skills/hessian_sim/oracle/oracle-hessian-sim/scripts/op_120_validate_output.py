"""Validate outputs/{case_id}.npz against the task contract.
Usage: python scripts/op_120_validate_output.py outputs/case_000.npz
Run from the eval run root with the run-local python.
"""
import sys
import numpy as np

EXPECTED_SHAPE = (12, 512, 512)
OK_DTYPES = (np.float32, np.float64)


def validate(path):
    with np.load(path) as z:
        if "data" not in z.files:
            raise SystemExit(f"FAIL {path}: missing array key 'data' (found {z.files})")
        arr = z["data"]
    if tuple(arr.shape) != EXPECTED_SHAPE:
        raise SystemExit(f"FAIL {path}: shape {arr.shape} != {EXPECTED_SHAPE}")
    if arr.dtype.type not in OK_DTYPES:
        raise SystemExit(f"FAIL {path}: dtype {arr.dtype} not in float32/float64")
    if not np.isfinite(arr).all():
        raise SystemExit(f"FAIL {path}: non-finite values present")
    print(f"OK {path}: shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min():.4g} max={arr.max():.4g}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/op_120_validate_output.py <npz> [<npz> ...]")
    for p in sys.argv[1:]:
        validate(p)
