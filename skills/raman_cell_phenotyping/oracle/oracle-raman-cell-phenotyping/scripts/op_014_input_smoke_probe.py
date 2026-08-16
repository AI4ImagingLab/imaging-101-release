#!/usr/bin/env python
"""Smoke probe for the Raman phenotyping input case.

Run from the eval run root:
    python ${CLAUDE_SKILL_DIR}/scripts/op_014_input_smoke_probe.py case_000
"""
import sys
import numpy as np


def main(case_id):
    base = f"public/visible_data/cases/{case_id}/input_data"
    d = np.load(f"{base}/raw_data.npz")
    vol = d["spectral_volume"]
    axis = d["spectral_axis"]
    print("spectral_volume:", vol.shape, vol.dtype)
    print("spectral_axis  :", axis.shape, axis.dtype)
    mask = (axis >= 700) & (axis <= 1800)
    n = int(mask.sum())
    print("fingerprint channels [700,1800]:", n)
    if n > 0:
        kept = axis[mask]
        print("cropped range cm-1:", float(kept[0]), float(kept[-1]))
    assert vol.ndim == 5 and vol.shape[1:4] == (40, 40, 10), "unexpected volume shape"
    assert axis.ndim == 1, "unexpected axis shape"
    print("OK: input smoke probe passed")


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "case_000"
    main(case)
