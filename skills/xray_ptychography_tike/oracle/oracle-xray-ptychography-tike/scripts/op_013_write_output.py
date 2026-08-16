#!/usr/bin/env python
"""Extract phase from a reconstructed object and write the submission npz.

Usage: python scripts/op_013_write_output.py <psi.npy> <output_dir>
psi.npy: complex array, (1, 482, 479) or (482, 479); writes
<output_dir>/case_000.npz containing 'object_phase'.
"""
import os
import sys

import numpy as np


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    psi_path, output_dir = sys.argv[1], sys.argv[2]
    psi = np.load(psi_path)
    phase = np.angle(psi[0] if psi.ndim == 3 else psi)
    assert phase.ndim == 2, phase.shape
    assert np.isfinite(phase).all(), "non-finite phase values"
    phase = np.asarray(phase, dtype=np.float32)
    if phase.shape != (482, 479):
        raise AssertionError(
            "expected (482, 479) object frame, got %s; align top-left and slice"
            % (phase.shape,))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "case_000.npz")
    np.savez(out_path, object_phase=phase)
    print("saved", out_path, phase.shape, phase.dtype)


if __name__ == "__main__":
    main()
