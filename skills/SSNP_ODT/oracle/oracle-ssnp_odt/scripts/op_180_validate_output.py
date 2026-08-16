"""Validate an SSNP-ODT submission NPZ against the output schema.

Usage (from eval run root):
    python scripts/op_180_validate_output.py outputs/case_000.npz
"""
import sys
import numpy as np

EXPECTED_SHAPE = (1, 308, 256, 256)
KEY = "delta_n"


def validate(path):
    errs = []
    with np.load(path) as z:
        if KEY not in z.files:
            errs.append("missing key '%s' (found %s)" % (KEY, z.files))
            return errs
        arr = z[KEY]
    if tuple(arr.shape) != EXPECTED_SHAPE:
        errs.append("shape %s != expected %s" % (tuple(arr.shape), EXPECTED_SHAPE))
    if arr.dtype not in (np.float32, np.float64):
        errs.append("dtype %s not in {float32, float64}" % arr.dtype)
    if not np.isfinite(arr).all():
        errs.append("array contains non-finite values")
    if np.isfinite(arr).all():
        print("  value range: [%.6g, %.6g]" % (float(arr.min()), float(arr.max())))
        if float(arr.min()) < -1e-6:
            print("  WARN: negative values present (non-negativity expected)")
    return errs


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/op_180_validate_output.py <path.npz>")
        sys.exit(2)
    path = sys.argv[1]
    errs = validate(path)
    if errs:
        print("INVALID:")
        for e in errs:
            print("  - " + e)
        sys.exit(1)
    print("VALID: %s" % path)


if __name__ == "__main__":
    main()
