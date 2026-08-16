#!/usr/bin/env python
"""Validate an FPM amplitude-stack submission npz.

Usage:
  public/envs/runtime/.venv-posix/bin/python \
      ${CLAUDE_SKILL_DIR}/scripts/op_120_validate_submission.py outputs/case_000.npz
"""
import sys
import numpy as np

EXPECTED_SHAPE = (1024, 1024, 161)
OK_DTYPES = ("float32", "float64")


def validate(path):
    errs = []
    try:
        data = np.load(path)
    except Exception as e:  # noqa: BLE001
        return [f"cannot load {path}: {e}"]
    if "amplitude_stack" not in data.files:
        return [f"missing key 'amplitude_stack' in {path}; keys={data.files}"]
    arr = data["amplitude_stack"]
    if tuple(arr.shape) != EXPECTED_SHAPE:
        errs.append(f"shape {arr.shape} != {EXPECTED_SHAPE}")
    if arr.dtype.name not in OK_DTYPES:
        errs.append(f"dtype {arr.dtype.name} not in {OK_DTYPES}")
    if not np.all(np.isfinite(arr)):
        errs.append("array contains non-finite values (NaN/Inf)")
    return errs


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/case_000.npz"
    errs = validate(path)
    if errs:
        print("FAIL:", path)
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print("OK:", path)


if __name__ == "__main__":
    main()
