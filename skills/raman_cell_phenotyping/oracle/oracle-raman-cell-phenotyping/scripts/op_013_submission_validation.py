#!/usr/bin/env python
"""Validate outputs/<case_id>.npz against the Raman phenotyping contract.

Run from the eval run root:
    python ${CLAUDE_SKILL_DIR}/scripts/op_013_submission_validation.py case_000
"""
import sys
import numpy as np

REQUIRED = [
    "abundance_lipids",
    "abundance_nucleus",
    "abundance_cytoplasm",
    "abundance_background",
]
OK_DTYPES = (np.float32, np.float64)


def main(case_id):
    path = f"outputs/{case_id}.npz"
    d = np.load(path)
    problems = []
    for k in REQUIRED:
        if k not in d:
            problems.append(f"missing key: {k}")
            continue
        a = d[k]
        if a.shape != (40, 40):
            problems.append(f"{k}: shape {a.shape} != (40, 40)")
        if a.dtype.type not in OK_DTYPES:
            problems.append(f"{k}: dtype {a.dtype} not float32/float64")
        if not np.all(np.isfinite(a)):
            problems.append(f"{k}: contains non-finite values")
    if problems:
        print("FAIL")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"OK: {path} passes all checks")


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "case_000"
    main(case)
