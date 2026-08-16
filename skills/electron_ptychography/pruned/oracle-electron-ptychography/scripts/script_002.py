"""Cheap smoke probe for the ptychography pipeline.

Run from the eval run root with run-local python, pointing at a case input dir:
    python ${CLAUDE_SKILL_DIR}/scripts/script_002.py <input_dir>
where <input_dir> contains raw_data.npz. Non-mutating; prints diagnostics only.
"""
import os
import sys
import numpy as np


def main(input_dir):
    raw = np.load(os.path.join(input_dir, "raw_data.npz"))
    dc = raw["datacube"][0]
    vp = raw["vacuum_probe"][0]
    print("datacube", dc.shape, dc.dtype, "probe", vp.shape)
    assert dc.ndim == 4 and vp.ndim == 2
    Rx, Ry, Qx, Qy = dc.shape

    # trivial probe: centered sqrt of vacuum intensity, no defocus (probe check only)
    P = np.sqrt(np.maximum(vp.astype(np.float64), 0)).astype(np.complex128)
    assert np.isfinite(P).all() and np.abs(P).sum() > 0

    Nx, Ny = Rx + Qx - 1, Ry + Qy - 1
    O = np.ones((Nx, Ny), dtype=np.complex128)
    amp = np.sqrt(np.maximum(dc.reshape(Rx * Ry, Qx, Qy).astype(np.float64), 0))
    mean_I = np.mean(dc.astype(np.float64) ** 0 * dc)  # mean intensity
    rng = np.random.default_rng(0)
    sub = rng.choice(Rx * Ry, size=32, replace=False)
    err0 = 0.0
    for j in sub:
        rx, ry = j // Ry, j % Ry
        patch = O[rx:rx + Qx, ry:ry + Qy]
        ff = np.fft.fft2(P * patch)
        err0 += np.sum((amp[j] - np.abs(ff)) ** 2)
    print("subset amplitude residual (finite):", np.isfinite(err0), float(err0))

    phase = np.angle(O[Qx // 4:Nx - Qx // 4, Qy // 4:Ny - Qy // 4])
    print("phase pre-resize shape", phase.shape, "finite", np.isfinite(phase).all())
    print("OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
