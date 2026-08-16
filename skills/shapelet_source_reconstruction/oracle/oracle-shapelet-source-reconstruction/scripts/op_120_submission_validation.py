#!/usr/bin/env python
"""Validate a shapelet source-reconstruction submission file.

Usage: python op_120_submission_validation.py outputs/<case_id>.npz
"""
import sys
import numpy as np


def validate(path):
    errs = []
    try:
        d = np.load(path)
    except Exception as e:  # noqa
        return ["cannot load %s: %r" % (path, e)]
    if 'source_image' not in d.files:
        errs.append("missing key 'source_image'")
        return errs
    a = d['source_image']
    if tuple(a.shape) != (320, 320):
        errs.append("shape %r != (320, 320)" % (tuple(a.shape),))
    if a.dtype not in (np.float32, np.float64):
        errs.append("dtype %s not float32/float64" % a.dtype)
    if not np.all(np.isfinite(a)):
        errs.append("non-finite values present")
    return errs


def main():
    if len(sys.argv) < 2:
        print("usage: op_120_submission_validation.py <npz> [<npz> ...]")
        sys.exit(2)
    bad = 0
    for p in sys.argv[1:]:
        errs = validate(p)
        if errs:
            bad += 1
            print("FAIL %s" % p)
            for e in errs:
                print("  - %s" % e)
        else:
            print("OK   %s" % p)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
